def build_prompt(saree_desc: str, hairstyle: str, lighting: str) -> str:
    """
    Builds a detailed Stable Diffusion prompt for saree fashion photography.
    """
    return (
        f"A beautiful Indian fashion model wearing {saree_desc}, "
        f"{hairstyle} hairstyle, "
        f"{lighting} lighting, "
        "realistic saree draping, standing gracefully, "
        "full body shot, elegant pose, "
        "ultra realistic, 4k, professional fashion photography, "
        "sharp focus, studio background, high detail"
    )
