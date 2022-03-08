from pydantic import BaseModel

class ArchConfig(BaseModel):
    """Generic base configuration for face detection architectures"""
    input_width: int
    input_height: int
    input_channel: int
    backbone: str
