def assemble_einops_string(input_shape: str, output_shape: str) -> str:
    """Assemble einops string"""
    return input_shape + " -> " + output_shape
