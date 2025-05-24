All completed except a small list of changes that will improve final script score.







Changes reduced the linting score to 9.3/10... just have to make following changes in list to take it to 10/10

List of things to clean next:

1. **Import-related issues** (R0402, C0415):
   - Use `from transformers.models.bert import modeling_bert` instead of `import transformers.models.bert.modeling_bert as modeling_bert`
   - Move imports to top level (no imports inside functions)
   - Avoid reimporting types (Dict, List)

2. **Function/Argument naming** (C0103):
   - Rename functions to follow snake_case:
     - `getAllLayers` → `get_all_layers`
     - `getAllLayersBert` → `get_all_layers_bert`
   - Rename arguments to follow snake_case:
     - `ALLROUNDSCLIENTS2CLASS` → `all_rounds_clients2class`

3. **Function arguments** (R0917):
   - Too many positional arguments in several functions (should be ≤ 5)
   - Need to refactor to use keyword arguments or data classes

4. **Logging format** (W1203):
   - Replace f-string logging with % formatting:
   ```python
   # Instead of:
   logging.info(f"client ids: {self.client_ids}")
   # Use:
   logging.info("client ids: %s", self.client_ids)
   ```

5. **Exception handling** (W0718):
   - Replace broad `except Exception` with specific exceptions
   - Add specific exception types for different error cases

6. **Attribute initialization** (W0201):
   - Move attribute definitions to `__init__`:
     - `global_neurons_inputs_outputs_batch`
     - `inputs2layer_grads`
     - `client2data`
     - `server_testdata`

7. **Unused arguments** (W0613):
   - Remove or use unused parameters:
     - `module` in hook functions
     - `test_data` in some functions
     - `config` in some functions

8. **Code style** (R1705, R1701):
   - Remove unnecessary `else` after `return`
   - Merge multiple `isinstance` checks
   - Fix line length (C0301)

9. **Import resolution** (unresolved imports):
   - These are not actual errors but environment setup issues
   - Need to install required packages: torch, transformers


