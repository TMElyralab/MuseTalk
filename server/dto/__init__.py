from pydantic import BaseModel, Field
from typing import Literal, Optional

class MuseTalkRequest(BaseModel):
    fps: Optional[int] = Field(default=25, description="Ouput video fps")
    avatar_id: str = Field(..., description="avatar id")
    skip_save_images: bool = Field(default=False, description="Whether skip saving images for better generation speed calculation")
    audio_path: str = Field(..., description="audio")
    ...

class AvatarRequest(BaseModel):
    video_path: str = Field(..., description="Driving pose video")
    batch_size: Optional[int] = Field(default=4, description="batch size")
    bbox_shift: Optional[int] = Field(default=5, description="We have found that upper-bound of the mask has an important impact on mouth openness. Thus, to control the mask region, we suggest using the bbox_shift parameter. Positive values (moving towards the lower half) increase mouth openness, while negative values (moving towards the upper half) decrease mouth openness.")
    