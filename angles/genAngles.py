import math

import cairosvg


def generate_svg_arrow(angle_deg, filename=None):
    # Configuration
    cx, cy = 92, 92  # Center coordinates
    L = 87  # Arrow length
    stroke_width = 3

    angle_rad = math.radians(angle_deg)
    offset_rad = math.radians(20)  # Arrowhead angle offset

    # Calculate main arrow coordinates
    x2 = cx + L * math.sin(angle_rad)
    y2 = cy - L * math.cos(angle_rad)  # Subtract because SVG Y increases downward

    # SVG template
    svg = f"""<svg width="184" height="92" viewBox="0 0 184 92" xmlns="http://www.w3.org/2000/svg">
  <g stroke="lightgray" stroke-width="4"
     stroke-linecap="round" stroke-linejoin="round" fill="none">
    <line x1="0" y1="90" x2="184" y2="90"/>
  </g>
    <g stroke="lightgray" stroke-width="4"
     stroke-linecap="round" stroke-linejoin="round" fill="none">
    <line x1="92" y1="4" x2="92" y2="90"/>
  </g>
    <g stroke="white" stroke-width="6"
     stroke-linecap="round" stroke-linejoin="round" fill="none">
    <line x1="{cx}" y1="{cy}" x2="{x2:.2f}" y2="{y2:.2f}"/>
  </g>
    <path d="M 2 91 A 62 60 0 0 1 182 90" fill="none" stroke="lightgray" stroke-width="4"/>
</svg>"""

    # Save to file if requested
    if filename:
        with open(filename, "w") as f:
            f.write(svg)
    return svg


# Generate all arrows from -90° to +60°
for angle in range(-90, 91):
    svg_content = generate_svg_arrow(angle, filename=f"angles/arrow_{angle:03d}.svg")
    cairosvg.svg2png(
        url=f"angles/arrow_{angle:03d}.svg", write_to=f"angles/arrow_{angle:03d}.png"
    )
