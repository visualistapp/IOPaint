from .power_paint.power_paint import PowerPaint
from .sd import SD15, SD2, Anything4, RealisticVision14, SD
from .sdxl import SDXL

models = {
    SD15.name: SD15,
    Anything4.name: Anything4,
    RealisticVision14.name: RealisticVision14,
    SD2.name: SD2,
    SDXL.name: SDXL,
    PowerPaint.name: PowerPaint,
}
