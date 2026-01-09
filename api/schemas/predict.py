from pydantic import BaseModel
from typing import Optional

class PredictInput(BaseModel):
	close: float
	ret: float
	acc: float
	amp: float
	vol_rel: float
	real_close_next: Optional[float] = None
	real_amp_next: Optional[float] = None

class PredictLiteInput(BaseModel):
	open: float
	close: float
	volume: float
	high: Optional[float] = None
	low: Optional[float] = None
