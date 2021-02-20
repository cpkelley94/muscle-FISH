# Example HCR-FISH analysis

This basic example illustrates how we use our general Python pipeline to analyze confocal microscopy images of myofibers. In this fiber, we used HCR-FISH to label mRNAs from the vinculin gene (Vcl), and we stained with DAPI to label nuclei:

<img src="vcl_channels.gif" alt="vcl_channels">

After opening the image and separating the channels, we first segment the myofiber from the background slide. This is accomplished by thresholding on the background signal in the FISH channel, using Li's method of automatic threshold selection.

<img src="vcl_fiber.gif" alt="vcl_fiber">