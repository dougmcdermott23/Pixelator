# Pixelator
Image filter that applies four unique filters to pixelate, smooth, limit palette size, and adjust saturation for a jpg or png image.

<img src="images/toronto.jpg" align="left" width="400" height="300" >
Base image
<br clear="left"/>

<img src="images/pixelate_toronto_filter_one.jpg" align="left" width="400" height="300" >
The first filter applies a mean filter to average groups of pixels and create super-samples. This gives the pixelated effect. Parameter "pixel_factor" sets the super-sample size (minimum of 1).
<br clear="left"/>

<img src="images/pixelate_toronto_filter_two.jpg" align="left" width="400" height="300" >
The second filter iterates through all super-samples and uses mean filtering to smooth the image. A super-sample cell colour is calculated as the average of itself and all surrounding neighbors. Parameter "smooth_iterations" sets the number of smooth steps performed on the image (0 to disable).
<br clear="left"/>

<img src="images/pixelate_toronto_filter_three.jpg" align="left" width="400" height="300" >
The third filter uses k-means clustering to limit the colour palette size. Parameter "palette_size" sets the number of clusters used (0 to disable).
<br clear="left"/>

<img src="images/pixelate_toronto_filter_four_over_sat.jpg" align="left" width="400" height="300" >
The fourth filter adjusts saturation level. Parameter "saturation" sets the saturation delta, set to +1 for maximum saturation and -1 for grey scale (0 to disable).
<br clear="left"/>

<img src="images/pixelate_toronto.jpg" width="400" align="left" height="300" >
An example of an image with the four above filters combined.
<br clear="left"/>
