#import libraries 
from copy import deepcopy
import numpy as np
import cv2
import random
import math 
from statistics import mode
import scipy.cluster.hierarchy as hcluster

#find minimum euclidean distance between three points
def min_dist(pt1, pt2, pt3):
  d1 = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
  d2 = np.sqrt((pt3[0] - pt2[0])**2 + (pt3[1] - pt2[1])**2)
  d3 = np.sqrt((pt1[0] - pt3[0])**2 + (pt1[1] - pt3[1])**2)
  return min(d1, d2, d3)


#get centre and radius of circle passing through 3 points 
def get_circle(pt1, pt2, pt3):

  #directly uses formula from solving for the centre and radius from three equations
  x1 = pt1[0]
  x2 = pt2[0]
  x3 = pt3[0]
  y1 = pt1[1]
  y2 = pt2[1]
  y3 = pt3[1]

  x12 = x1 - x2
  x13 = x1 - x3 
  y12 = y1 - y2
  y13 = y1 - y3 
  y31 = y3 - y1
  y21 = y2 - y1 
  x31 = x3 - x1
  x21 = x2 - x1
 
  sx13 = pow(x1, 2) - pow(x3, 2)
  sy13 = pow(y1, 2) - pow(y3, 2) 
  sx21 = pow(x2, 2) - pow(x1, 2);
  sy21 = pow(y2, 2) - pow(y1, 2);
 
  f = (((sx13) * (x12) + (sy13) *
        (x12) + (sx21) * (x13) +
        (sy21) * (x13)) // (2 *
        ((y31) * (x12) - (y21) * (x13))))
             
  g = (((sx13) * (y12) + (sy13) * (y12) +
        (sx21) * (y13) + (sy21) * (y13)) //
        (2 * ((x31) * (y12) - (x21) * (y13))))

  return int(-g), int(-f)


#get points in two separate connected components(concentric circles here) from image
def get_points(edges):
  #diving image into contours
  ret, thresh = cv2.threshold(edges, 127, 255, 0)
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea, reverse = True)

  #getting contour with maximum area, taking it to be points from the bigger circle
  area_big = cv2.contourArea(contours[0])
  big_points = [tuple(i) for i in np.vstack(contours[0]).squeeze()]

  area_diff = 100000
  area = area_big

  #moving in descending order of contour area, 
  #selecting contour of points from the smaller circle such that it's area is more than some fixed difference less
  i = 1
  while(area_big - area < area_diff):
      area = cv2.contourArea(contours[i])
      i = i + 1

  small_points = [tuple(i) for i in np.vstack(contours[i]).squeeze()]

  
  return big_points, small_points


#get centre from given connected component
def get_centre(points, defect):
  #Select 3 points from connected component randomly. Find centre of circle made by points 
  #Repeat for some iterations. Choose centre with most votes

  sample_points = [0, 0, 0]

  #initialise accumulator array for x and y coordinates of centre
  centre_arr_x = np.zeros(defect.shape[1], dtype="uint8")
  centre_arr_y = np.zeros(defect.shape[0], dtype="uint8")

  #run sampling for points for some iterations 
  iters = 20
  min_dist_ = 50
  for i in range(iters):
    d = 0
    #sample 3 points which are separated by more than some minimum distance
    while(d < min_dist_):
      sample_points = random.sample(points, 3)
      d = min_dist(sample_points[0], sample_points[1], sample_points[2])


    len_x = len(centre_arr_x)
    len_y = len(centre_arr_y)
    centre = [len_x, len_y]

    #centre out of bounds error control
    while (centre[0] >= len_x) or (centre[1] >= len_y):
      centre = get_circle(sample_points[0], sample_points[1], sample_points[2])

    #increment accumulator array
    centre_arr_x[centre[0]] = centre_arr_x[centre[0]] + 1 
    centre_arr_y[centre[1]] = centre_arr_y[centre[1]] + 1 

  #return centre with maximum votes
  max_centres_x = np.where(centre_arr_x == centre_arr_x.max())
  max_centres_y = np.where(centre_arr_y == centre_arr_y.max())
  return (int(max_centres_x[0][0]), int(max_centres_y[0][0]))


#get radius of a set of points with given centre
def get_radius(centre, points):
  #calculating distance from centre of randomly sampled points, returning the most frequent
  radius = []
  for i in range(50):
    point = random.sample(points, 1)[0]
    dist = np.sqrt((point[0] - centre[0])**2 + (point[1] - centre[1])**2)
    radius.append(int(dist))
    return mode(radius)

#get centre and radii of two approximate circles given defect image
def circle_detection(defect):
#canny edge detection
  edges = cv2.Canny(defect,100,200)
  #get connected components
  big_points, small_points = get_points(edges)

  #get centre of concentric circles
  centre = get_centre(small_points, defect)

  #get radius of the circles
  small_radius = get_radius(centre, small_points)
  big_radius = get_radius(centre, big_points)

  return centre, small_radius, big_radius


#given the substracted image, differentiates the defect from the circle slivers
#returns coordinates of points with defect 
def localise_defects(img, centre, radius, diff_threshold, granularity, rect_size):
  centre_ = [0, 0]
  start = [0, 0]
  end = [0, 0]
  area_defects = []
  output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  #loop through small regions on the edge of the circle
  #store the number of white pixels(area) of each region 
  for theta in range(0, 360, granularity):
    theta_ = (theta + granularity/2)*2*math.pi/360
    centre_[0] = centre[0] + int(radius*math.cos(theta_))
    centre_[1] = centre[1] + int(radius*math.sin(theta_))
    start[0] = centre_[0] - rect_size
    start[1] = centre_[1] - rect_size
    end[0] = centre_[0] + rect_size
    end[1] = centre_[1] + rect_size
    cv2.rectangle(output, start, end, (255, 0, 0), 2)
    rect = img[start[1]:end[1], start[0]:end[0]]
    area = np.sum(rect == 255)
    area_defects.append((area, centre_[0], centre_[1]))

  #selecting regions which have significantly more white pixels than neighbouring regions 
  bbox = []
  for i in range(len(area_defects)):
    if ((area_defects[i][0] - area_defects[i - 3][0]) > diff_threshold) and ((area_defects[i][0] - area_defects[(i + 3)%len(area_defects)][0]) > diff_threshold):
      bbox.append(area_defects[i])

  return bbox

#returns average of each cluster, clustered based on distance
def cluster_points(bbox, thresh):
  #use
  bbox_ = []
  len_b = len(bbox)
  if len_b <= 1:
    return bbox
  else:
    clusters = hcluster.fclusterdata(bbox, thresh, criterion="distance")
    groups = np.unique(clusters)
    for group in groups:
      bbox_group_x = [bbox[i][0] for i in range(len_b) if clusters[i] == group]
      bbox_group_y = [bbox[i][1] for i in range(len_b) if clusters[i] == group]
      bbox_.append([int(np.average(bbox_group_x)), int(np.average(bbox_group_y))])
    return bbox_

#########################################################################################################################################################################################
#driver code starts here

#some assumed constants
rect_size = 30
granularity_big = 2
granularity_small = 5
diff_threshold = 150
pooling_size = 4
distance_thresh = 10

#read all defective images
defect1 = cv2.imread("images/defect1.png", 0)
defect2 = cv2.imread("images/defect2.png", 0)
defect3 = cv2.imread("images/defect3.png", 0)
defect4 = cv2.imread("images/defect4.png", 0)

#running algorithm on all given defective images
for defect in [defect1, defect2, defect3, defect4]:
  #creating accumulator array for voting on localised defect locations
  localisation_arr = np.zeros([int(defect.shape[0]/pooling_size), int(defect.shape[1]/pooling_size)], dtype = "int")

  #iterating 20 times over main algorithm and voting on defect location to reduce random error
  iters = 20
  for i in range(iters):
    #getting the centre and two radii of the best fit circles
    centre, small_radius, big_radius = circle_detection(defect)

    #get an image similar to the given good image, putting the two concentric circles in a white background
    recreate = np.full(defect.shape, 255, dtype='uint8')
    cv2.circle(recreate, centre, big_radius, 0, -1)
    cv2.circle(recreate, centre, small_radius, 255, -1)

    #substract the created image from defect image and round negative numbers to 0
    #to get rough idea of location of cuts(areas where the created image is black but the defect image is white)
    cuts = cv2.subtract(defect, recreate)

    #substract the defect image from created image and round negative numbers to 0
    #to get rough idea of location of flashes(areas where the created image is white but the defect image is black)
    flashes = cv2.subtract(recreate, defect)

    #get the coordinates location of defects
    localised_flashes_small = localise_defects(flashes, centre, small_radius, diff_threshold, granularity_small, rect_size)
    localised_flashes_big =  localise_defects(flashes, centre, big_radius, diff_threshold, granularity_big, rect_size)

    localised_cuts_small = localise_defects(cuts, centre, small_radius, diff_threshold, granularity_small, rect_size)
    localised_cuts_big = localise_defects(cuts, centre, big_radius, diff_threshold, granularity_big, rect_size)

    #increment accumulator array on the location of cuts
    cut_ = [0, 0]
    for cut in localised_cuts_small:
      cut_[0] = int(cut[1]/pooling_size)
      cut_[1] = int(cut[2]/pooling_size)
      localisation_arr[cut_[0]][int(cut_[1])] = localisation_arr[cut_[0]][cut_[1]] + 1
    for cut in localised_cuts_big:
      cut_[0] = int(cut[1]/pooling_size)
      cut_[1] = int(cut[2]/pooling_size)
      localisation_arr[cut_[0]][int(cut_[1])] = localisation_arr[cut_[0]][cut_[1]] + 1

    flash_ = [0, 0]
    for flash in localised_flashes_small:
      flash_[0] = int(flash[1]/pooling_size)
      flash_[1] = int(flash[2]/pooling_size)
      localisation_arr[flash_[0]][flash_[1]] = localisation_arr[flash_[0]][flash_[1]] - 1
    for flash in localised_flashes_big:
      flash_[0] = int(flash[1]/pooling_size)
      flash_[1] = int(flash[2]/pooling_size)
      localisation_arr[flash_[0]][flash_[1]] = localisation_arr[flash_[0]][flash_[1]] - 1


  #choose all the points in accumulator array which have more than equal to some number of votes
  num_votes = int(iters/6)
  bbox_cuts = np.argwhere(localisation_arr >= num_votes)
  bbox_flashes = np.argwhere(localisation_arr <= -num_votes)

  output = cv2.cvtColor(defect, cv2.COLOR_GRAY2BGR)
  for cut in bbox_cuts:
    cv2.rectangle(output, [cut[0]*pooling_size - rect_size, cut[1]*pooling_size - rect_size], [cut[0]*pooling_size + rect_size, cut[1]*pooling_size + rect_size], (0, 255, 0), 2)
  for cut in bbox_flashes:
    cv2.rectangle(output, [cut[0]*pooling_size - rect_size, cut[1]*pooling_size - rect_size], [cut[0]*pooling_size + rect_size, cut[1]*pooling_size + rect_size], (0, 255, 0), 2)   

  #cluster points based on distance with each other, store average for each cluster
  bbox_cuts = cluster_points(bbox_cuts, distance_thresh)
  bbox_flashes = cluster_points(bbox_flashes, distance_thresh)

  #plot localised defects in image
  #GREEN for flashes
  #BLUE for cuts
  defect = cv2.cvtColor(defect, cv2.COLOR_GRAY2BGR)
  for cut in bbox_cuts:
    cv2.rectangle(defect, [cut[0]*pooling_size - rect_size, cut[1]*pooling_size - rect_size], [cut[0]*pooling_size + rect_size, cut[1]*pooling_size + rect_size], (255, 0, 0), 2)

  for flash in bbox_flashes:
    cv2.rectangle(defect, [flash[0]*pooling_size - rect_size, flash[1]*pooling_size - rect_size], [flash[0]*pooling_size + rect_size, flash[1]*pooling_size + rect_size], (0, 255, 0), 2)

  cv2.imshow("defects localised.jpg", defect)
  cv2.waitKey(0)
cv2.destroyAllWindows() 