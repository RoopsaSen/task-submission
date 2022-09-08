Algorithm:
Part 1: Recreating original image
1. Detect edges of circles using canny edge detection
2. Detect two sets of points for small and large circles using contour detection in cv2
3. Sample 3 points in a contour, find the center of the circle passing through these points by solving 3 equations. Vote on the center after some iterations.
4. For found center take the maximum occurring value of radius for each set of points
[Hough circle was not used as keeping the radius as variable would add a lot of extra computation, considering the fact that only 2 radii are possible. Also for optimisation purposes, the pre-built hough circle function is not that good in detecting concentric circles]
5. Recreate a good image with approximated circles

Part 2: Differentiating between cuts and flashes
1. It can be seen that cuts in the image are areas which are black in the good image and white in the defective image. Similarly flashes are the areas where the good image is white while the defective image is black. Taking advantage of that, the pixel wise values of the good image from defective(for cuts) have been subtracted. Opposite is done for the flashes

Part 3: Localizing the defects
1. As the circle detection is not perfect, there are extra parts in the cut and flash images. However the defects can be differentiated as they have much more localized white pixels than their neighboring areas. Thus areas around the circumference of the circle are sampled and areas with a higher number of white pixels than their neighboring area(over a taken threshold) are taken.
[This method will not work if the defects are larger than what is shown in the given images]
3. Points close to each other are clustered using hierarchical clustering. 

Comments:
After iterating enough times and voting for defects, the current algorithm gives the correct result for all images.
However it sometimes cannot detect the cut in defect2, probably because it is quite small and the approximated circle falls beyond it.
