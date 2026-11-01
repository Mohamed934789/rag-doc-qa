"""Generate a fictional PDF owned by the user for local demonstration."""
from pathlib import Path
from reportlab.pdfgen import canvas
if __name__ == "__main__":
    target = Path("data/sample.pdf"); target.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(target)); c.drawString(40, 800, "Fictional garden guide: water orchids once a week.")
    c.drawString(40, 780, "Keep orchids in indirect sunlight. Synthetic demonstration data.")
    c.save(); print(target)
