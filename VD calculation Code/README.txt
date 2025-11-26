# Vessel Density (VD) calculation Code
"A semi-automated algorithm using Matlab Software"

The mentioned code is implementeted using Matlab software R2019a (Mathworks, Inc., Natick, MA). 
Our cotributions for VD calculation are threefold. Firstly, preprocessing stage was applied for vessel enhancement and noise reduction.
An Otsu algorithm was subsequently applied for detection of retinal vessels in superficial and deep capillary plexus. 
Finally, vessel area density and vessel skeleton density were calculated by dividing the number of pixels associated with capillaries over the number of pixels in the entire image.

In order to use this algorithm you firt need to install Matlab software and run "Vessel_Density_Calculation_Code.m"
Loading a relevant data, the user should set the center of macula (fovea) using a click in this area.In this case, an interactive polygon
create using move the mouse and click within foveal avascular zone (FAZ) to used in noise reduction stage.
Then code automatically calculated VD and saved the results.  

