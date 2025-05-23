"""fls.py."""

import gc
import logging
import numpy as np
import time

import torch

from flwr.common import ndarrays_to_parameters
from tracefl.fl_provenance import round_lambda_prov
from tracefl.models_utils import (
    get_parameters,
    initialize_model,
    set_parameters,
)
from tracefl.models_train_eval import global_model_eval
from tracefl.strategy import FedAvgSave
from tracefl.utils import get_backend_config
from tracefl.dp_utils import safe_clip_inputs_inplace
from tracefl.dp_strategy import TraceFLDifferentialPrivacy


class FLSimulation:
    """Main class to run the simulation."""

    def __init__(self, cfg, ff, nsr, le, min_ev):
        self.all_rounds_results = []
        self.cfg = cfg
        self.strategy = None
        self.device = torch.device(self.cfg.tool.tracefl.device.device)
        self.backend_config = get_backend_config(cfg)
        self.fraction_fit = ff
        self.num_server_rounds = nsr
        self.local_epochs = le
        self.min_eval = min_ev
        self.start_time = time.time()

    def set_server_data(self, server_data):
        """Set the server's dataset for evaluation.

        Args:
            server_data: Dataset to be used for server-side evaluation
        """
        self.server_testdata = server_data

    def set_clients_data(self, clients_data):
        """Set the datasets for all clients.

        Args:
            clients_data: Dictionary mapping client IDs to their datasets
        """
        self.client2data = clients_data
        logging.info(f"Number of clients: {len(self.client2data)}")
        logging.info(f"Participating clients ids: {list(self.client2data.keys())}")
        if len(self.client2data) != self.cfg.tool.tracefl.data_dist.num_clients:
            self.cfg.tool.tracefl.num_clients = len(self.client2data)
            logging.warning(
                f"Adjusting number of clients to: {self.cfg.tool.tracefl.num_clients}"
            )

    def set_strategy(self):
        """Set the federated learning strategy.

        This method configures the FL strategy based on the configuration.
        If differential privacy is enabled (noise_multiplier and clipping_norm > 0),
        it wraps the base strategy with a DP strategy.
        """
        try:
            model_dict = initialize_model(
                self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
            )
            initial_parameters = ndarrays_to_parameters(get_parameters(model_dict["model"]))

            # Verify initial parameters are not all zeros
            params = get_parameters(model_dict["model"])
            if all(np.all(p == 0) for p in params):
                logging.warning("Initial parameters are all zeros. Reinitializing model...")
                model_dict = initialize_model(
                    self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
                )
                initial_parameters = ndarrays_to_parameters(get_parameters(model_dict["model"]))

            strategy = FedAvgSave(
                initial_parameters=initial_parameters,
                cfg=self.cfg,
                accept_failures=False,
                fraction_fit=self.fraction_fit,
                fraction_evaluate=0,
                min_fit_clients=self.cfg.tool.tracefl.strategy.clients_per_round,
                min_evaluate_clients=0,
                min_available_clients=self.cfg.tool.tracefl.data_dist.num_clients,
                evaluate_fn=self._evaluate_global_model,
                evaluate_metrics_aggregation_fn=lambda metrics: {},
                on_fit_config_fn=self._get_fit_config,
                fit_metrics_aggregation_fn=self._fit_metrics_aggregation_fn,
            )

            # Apply differential privacy if enabled in config
            if (self.cfg.tool.tracefl.strategy.noise_multiplier > 0 and 
                self.cfg.tool.tracefl.strategy.clipping_norm > 0):
                logging.info(
                    ">> ----------------------------- Running DP FL -----------------------------"
                )
                dp_strategy = TraceFLDifferentialPrivacy(
                    strategy=strategy,
                    noise_multiplier=self.cfg.tool.tracefl.strategy.noise_multiplier,
                    clipping_norm=self.cfg.tool.tracefl.strategy.clipping_norm,
                    num_sampled_clients=self.cfg.tool.tracefl.strategy.clients_per_round,
                )
                self.strategy = dp_strategy
                logging.info(
                    f"Differential Privacy enabled with noise_multiplier={self.cfg.tool.tracefl.strategy.noise_multiplier}, "
                    f"clipping_norm={self.cfg.tool.tracefl.strategy.clipping_norm}"
                )
            else:
                logging.info(
                    ">> ----------------------------- Running Non-DP FL -----------------------------"
                )
                self.strategy = strategy

        except Exception as e:
            logging.error(f"Error setting up strategy: {str(e)}")
            raise

    def _fit_metrics_aggregation_fn(self, metrics):
        logging.info(">>   ------------------- Clients Metrics ------------- ")
        for nk, m in metrics:
            cid = int(m["cid"])
            logging.info(
                f" Client {cid}, Loss Train {m['train_loss']}, Accuracy Train {m['train_accuracy']}, data_points = {nk}"
            )
        return {"loss": 0.1, "accuracy": 0.2}

    def _get_fit_config(self, server_round: int):
        """Get configuration for client training."""
        torch.Generator().manual_seed(server_round)
        config = {
            "server_round": server_round,
            "local_epochs": self.cfg.tool.tracefl.client.epochs,
            "batch_size": self.cfg.tool.tracefl.data_dist.batch_size,
            "lr": self.cfg.tool.tracefl.client.lr,
            "grad_clip": 1.0  # Add gradient clipping threshold
        }
        return config

    def _evaluate_global_model(self, server_round, parameters, config):
        logging.info("Evaluating initial global parameters")
        try:
            model_dict = initialize_model(
                self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
            )
            set_parameters(model_dict["model"], parameters)
            model_dict["model"].eval()

            metrics = global_model_eval(
                self.cfg.tool.tracefl.model.arch, model_dict, self.server_testdata
            )
            loss = metrics["loss"]
            acc = metrics["accuracy"]
            self.all_rounds_results.append({"loss": loss, "accuracy": acc})

            if server_round == 0:
                logging.info(f"initial parameters (loss, other metrics): {loss}, {{'accuracy': {acc}, 'loss': {loss}, 'round': {server_round}}}")
                return loss, {"accuracy": acc, "loss": loss, "round": server_round}

            logging.info(f"fit progress: ({server_round}, {loss}, {{'accuracy': {acc}, 'loss': {loss}, 'round': {server_round}}}, {time.time() - self.start_time})")

            if self.strategy is None:
                logging.error("Strategy is not initialized")
                return loss, {
                    "accuracy": acc,
                    "loss": loss,
                    "round": server_round,
                    "error": "Strategy not initialized",
                }

            # Check if we're using a DP strategy and extract the inner strategy if so
            fedavg = self.strategy
            if isinstance(self.strategy, TraceFLDifferentialPrivacy):
                fedavg = self.strategy.strategy

            if not isinstance(fedavg, FedAvgSave):
                logging.error("Invalid strategy type")
                return loss, {
                    "accuracy": acc,
                    "loss": loss,
                    "round": server_round,
                    "error": "Invalid strategy type",
                }

            client2model = {}
            for cid, weights in fedavg.client2ws.items():
                m_dict = initialize_model(
                    self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
                )
                model = m_dict["model"]
                if weights is not None:
                    model.load_state_dict(weights)
                model.eval()
                client2model[cid] = model

            if not hasattr(fedavg, "gm_ws"):
                print(
                    f"gm_ws not set — skipping provenance analysis "
                    f"for round {server_round}"
                )
                return loss, {"accuracy": acc, "loss": loss, "round": server_round}

            prov_global_model_dict = initialize_model(
                self.cfg.tool.tracefl.model.name, self.cfg.tool.tracefl.dataset
            )
            prov_global_model = prov_global_model_dict["model"]
            prov_global_model.load_state_dict(fedavg.gm_ws)
            prov_global_model.eval()

            provenance_input = {
                "train_cfg": self.cfg.tool.tracefl,
                "prov_cfg": self.cfg,
                "prov_global_model": prov_global_model,
                "client2model": client2model,
                "client2num_examples": fedavg.client2num_examples,
                "ALLROUNDSCLIENTS2CLASS": fedavg.client2class,
                "central_test_data": self.server_testdata,
                "server_round": server_round,
            }

            logging.info(">> Running provenance analysis...")
            prov_result = round_lambda_prov(**provenance_input)
            logging.info(f">> Provenance analysis completed. Results:\n{prov_result}")

            gc.collect()
            return loss, {
                "accuracy": acc,
                "loss": loss,
                "round": server_round,
                "prov_result": prov_result,
            }

        except Exception as e:
            logging.error(
                f"Evaluation failed during round {server_round}: {str(e)}",
                exc_info=True,
            )
            return loss, {
                "accuracy": 0.0,
                "loss": loss,
                "round": server_round,
                "error": str(e),
            }
