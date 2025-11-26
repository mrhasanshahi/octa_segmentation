FAZ Segmentation – Semi-Automated MATLAB Script
-----------------------------------------------

This script (run_faz.m) performs semi-automated segmentation of the Foveal Avascular Zone (FAZ)
from OCTA images. It replicates the workflow of the original GUI version, where the technician
or doctor can adjust the segmentation threshold while viewing a live preview.

How it works:
1. Run the script in MATLAB.
2. Select an OCTA image when prompted.
3. The script preprocesses the image, segments the FAZ, and shows a 2×2 preview:
      - Input image
      - Enhanced image with FAZ boundary
      - Filled FAZ region
      - Max diameter + bounding box
4. If the segmentation is not satisfactory, the user can enter a new threshold value.
5. After confirming the segmentation, the script prints the FAZ measurements.

Extracted FAZ Features:
- FAZ area (mm²)
- Perimeter (mm)
- Maximum diameter (mm)
- Form factor
- Roundness
- Extent
- Convexity
- Solidity
- Axial ratio
- Border irregularity

Notes:
- This is a simplified version of the original App Designer tool.
- The segmentation is semi-automated: the user can adjust threshold values.
- No vessel density analysis is included.
- Requires MATLAB + Image Processing Toolbox.

Usage:
1. Add the script to your MATLAB path.
2. Run:  run_faz
3. Select an OCTA image.
4. Adjust threshold if needed.
