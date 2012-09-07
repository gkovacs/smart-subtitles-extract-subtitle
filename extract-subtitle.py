#!/usr/bin/python

import cv
from cv import *

import collections
import functools
import itertools

activation_threshold = 0.5
import sys
vidf = 'video.m4v'
if len(sys.argv) > 1:
  vidf = sys.argv[1]

class memoized(object):
  '''Decorator. Caches a function's return value each time it is called.
  If called later with the same arguments, the cached value is returned
  (not reevaluated).
  '''
  def __init__(self, func):
    self.func = func
    self.cache = {}
  def __call__(self, *args):
    if not isinstance(args, collections.Hashable):
      # uncacheable. a list, for instance.
      # better to not cache than blow up.
      return self.func(*args)
    if args in self.cache:
      return self.cache[args]
    else:
      value = self.func(*args)
      self.cache[args] = value
      return value
  def __repr__(self):
    '''Return the function's docstring.'''
    return self.func.__doc__
  def __get__(self, obj, objtype):
    '''Support instance methods.'''
    return functools.partial(self.__call__, obj)

def reducedColor(rgb):
  return tuple([round(x/8.0)*8.0 for x in rgb])

def colorMatch(color1, color2):
  abs_diff_sum = sum([abs(c1-c2) for c1,c2 in zip(color1,color2)])
  return (abs_diff_sum < 120)

def roughColorMatch(color1, color2):
  abs_diff_sum = sum([abs(c1-c2) for c1,c2 in zip(color1,color2)])
  return (abs_diff_sum < 200)

#def greatlyReducedColor(rgb):
#S  return tuple([round(x/50.0)*50.0 for x in rgb])

def getHarris(img):
  yuv = CreateImage(GetSize(img), 8, 3)
  gray = CreateImage(GetSize(img), 8, 1)
  CvtColor(img, yuv, CV_BGR2YCrCb)
  Split(yuv, gray, None, None, None)
  harris = CreateImage (GetSize(img), IPL_DEPTH_32F, 1)
  #CornerHarris(gray, harris, 9, 9, 0.1)
  CornerHarris(gray, harris, 7, 7, 0.1)
  return harris

def getCanny(img):
  yuv = CreateImage(GetSize(img), 8, 3)
  gray = CreateImage(GetSize(img), 8, 1)
  CvtColor(img, yuv, CV_BGR2YCrCb)
  Split(yuv, gray, None, None, None)
  canny = cv.CreateImage(cv.GetSize(img), 8, 1)
  cv.Canny(gray, canny, 50, 200)
  cv.SaveImage('canny.png', canny)
  return canny

def iterImg(img):
  for y in range(img.height):
    for x in range(img.width):
      yield y,x

def removeUnactivated(img, activation):
  output = CreateImage(GetSize(img), 8, 3)
  for y,x in iterImg(img):
    if activation[y,x] >= activation_threshold:
      output[y,x] = img[y,x]
    else:
      output[y,x] = (0,0,0)
  return output

def closestPoint(img, point):
  y,x = point
  if y >= img.height:
    y = img.height - 1
  if x >= img.width:
    x = img.width - 1
  if y < 0:
    y = 0
  if x < 0:
    x = 0
  return y,x

def iterImgSurrounding4(img):
  for y,x in iterImg(img):
    mid = y,x
    left = closestPoint(img, (y,x-1))
    right = closestPoint(img, (y,x-1))
    up = closestPoint(img, (y+1,x))
    down = closestPoint(img, (y-1,x))
    yield mid,left,right,up,down

def iterImgSurrounding8(img):
  for y,x in iterImg(img):
    mid = y,x
    left = closestPoint(img, (y,x-1))
    right = closestPoint(img, (y,x-1))
    up = closestPoint(img, (y+1,x))
    down = closestPoint(img, (y-1,x))
    leftup = closestPoint(img, (y+1,x-1))
    leftdown = closestPoint(img, (y-1,x-1))
    rightup = closestPoint(img, (y+1,x+1))
    rightdown = closestPoint(img, (y-1,x+1))
    yield mid,left,right,up,down,leftup,leftdown,rightup,rightdown

def rowSum(img, rownum):
  return sum([img[rownum,x] for x in range(img.width)])

def colSum(img, colnum):
  return sum([img[y,colnum] for y in range(img.height)])


num_portions = 16

@memoized
def getBestPortion(videofile):
  harrisSum = None
  max_portion_counts = [0.0]*num_portions
  for idx,img in iterVideo(videofile):
    if idx % 100 != 0:
      continue
    if not img:
      break
    img = getBottomQuarter(img)
    harris = getHarris(img)
    vals_and_portion = []
    for i in range(num_portions):
      rsum = sum([rowSum(harris, rownum) for rownum in range(i*img.height/num_portions,(i+1)*img.height/num_portions)])
      vals_and_portion.append((rsum, i))
    best_portion_val,best_portion = max(vals_and_portion)
    max_portion_counts[best_portion] += best_portion_val
    if idx >= 1000:
     break
  best_portion = max([(x,i) for i,x in enumerate(max_portion_counts)])[1]
  return best_portion

def addColorToHistogram(color, histogram):
  color = reducedColor(color)
  if color == (0,0,0):
    return
  if sum(color) < 200:
    return
  if color not in histogram:
    histogram[color] = 1
  else:
    histogram[color] += 1

def addImageToColorHistogram(img, histogram):
  for y,x in iterImg(img):
    addColorToHistogram(img[y,x], histogram)

def solidColorImg(color):
  img = CreateImage((1,1), 8, 3)
  for y,x in iterImg(img):
    img[y,x] = color
  return img

def iterVideo(videofile):
  vid = cv.CaptureFromFile(videofile)
  for idx in itertools.count(0):
    img = cv.QueryFrame(vid)
    if not img:
      break
    yield idx,img

def getBottomQuarter(img):
  sub = cv.GetSubRect(img, (0, img.height*3/4, img.width, img.height/4))
  return cv.GetMat(sub)

def getPortion(img, portion_num):
  sub = cv.GetSubRect(img, (0, img.height*portion_num/num_portions, img.width, img.height/num_portions))
  return cv.GetMat(sub)

def getCenterHorizontal(img):
  sub = cv.GetSubRect(img, (img.width/3, 0, img.width/3, img.height))
  return cv.GetMat(sub)

def isSubtitleFrame(harris):
  max_val = harris.width * harris.height
  cur_val = 0
  for y,x in iterImg(harris):
    if harris[y,x] >= activation_threshold:
      cur_val += 1
  if cur_val*4 > max_val:
    return True
  return False

@memoized
def getSubtitleColor(videofile):
  best_portion = getBestPortion(videofile)
  print best_portion
  color_histogram = {}
  for idx,img in iterVideo(videofile):
    if idx % 100 != 0:
      continue
    img = getBottomQuarter(img)
    img = getPortion(img, best_portion)
    img = getCenterHorizontal(img)
    harris = getHarris(img)
    if not isSubtitleFrame(harris):
      continue
    img_intersected = removeUnactivated(img, harris)
    #SaveImage(str(idx)+'.png', img)
    #SaveImage(str(idx)+'-harris.png', harris)
    #SaveImage(str(idx)+'-intersected.png', img_intersected)
    addImageToColorHistogram(img_intersected, color_histogram)
  colorcounts = [(count,color) for color,count in color_histogram.iteritems()]
  colorcounts.sort()
  colorcounts.reverse()
  print [(color,count) for count,color in colorcounts[0:4]]
  count,color = max(colorcounts)
  print count, color
  img = solidColorImg(color)
  SaveImage('solidcolor.png', img)
  return color

def extractColor(origimg, color):
  img = CreateImage((origimg.width, origimg.height), 8, 3)
  #for y,x in iterImg(img):
  for surrounding in iterImgSurrounding8(img):
    haveMatch = False
    centery,centerx = surrounding[0]
    if roughColorMatch(origimg[centery,centerx], color):
      for y,x in surrounding[1:]:
        if colorMatch(origimg[y,x], color):
          haveMatch = True
          img[centery,centerx] = origimg[centery,centerx]
          break
    if not haveMatch:
      img[centery,centerx] = (0,0,0)
  return img

subtitle_color = getSubtitleColor(vidf)
best_portion = getBestPortion(vidf)
'''
for idx,img in iterVideo(vidf):
  img = getBottomQuarter(img)
  img = getPortion(img, best_portion)
  SaveImage(str(idx)+'.png', img)
'''

subtitle_color = getSubtitleColor(vidf)
for idx,img in iterVideo(vidf):
  img = getBottomQuarter(img)
  extracted_color_img = extractColor(img, subtitle_color)
  SaveImage(str(idx)+'.png', extracted_color_img)

'''
  if not harrisSum:
   harrisSum = CreateImage (GetSize(img), IPL_DEPTH_32F, 1)
   Zero(harrisSum)
  harrisSumTemp = CreateImage (GetSize(img), IPL_DEPTH_32F, 1)
  Zero(harrisSumTemp)
  Add(harris, harrisSum, harrisSumTemp)
  harrisSum = harrisSumTemp
  scaledHarrisSum = CreateImage (GetSize(img), IPL_DEPTH_32F, 1)
  Zero(scaledHarrisSum)
  Scale(harrisSum, scaledHarrisSum, 1.0/(idx+1))
  SaveImage(str(idx)+'.png', scaledHarrisSum)
  idx += 1
'''





#SaveImage('harris.png', harris)

#for y in range(harris.rows):
#  

"""
vid = cv.CaptureFromFile('video.m4v')
# determine color of subtitles
framenum = 0
motion_stripped = None
while True:
  frame = cv.QueryFrame(vid)
  if not frame:
   break
  sub = cv.GetSubRect(frame, (0, frame.height*3/4, frame.width, frame.height/4))
  mat = cv.GetMat(sub)
  downsampled = cv.CreateMat(mat.rows / 2, mat.cols / 2, cv.CV_8UC3)
  print mat.type
  print downsampled.type
  cv.Resize(mat, downsampled)
  mat = downsampled
  color_changes = {}
  for y in range(mat.rows):
   prev_color = (0.0,0.0,0.0)
   for x in range(mat.cols):
    cur_color = reducedColor(mat[y,x])
    if cur_color != prev_color and (max(cur_color) > 80):
      if cur_color not in color_changes:
       color_changes[cur_color] = 1
      else:
       color_changes[cur_color] += 1
  max_change_color = max([(v,k) for k,v in color_changes.iteritems()])[1]
  max_change_color_img = cv.CreateMat(1,1,cv.CV_8UC3)
  max_change_color_img[0,0] = max_change_color
  cv.SaveImage(str(framenum) + '.png', max_change_color_img)
  framenum += 1
  '''
  if framenum % 5 == 0:
   if motion_stripped:
    cv.SaveImage(str(framenum/5) + '.png', motion_stripped)
   motion_stripped = cv.CreateMat(mat.rows, mat.cols, cv.CV_8UC3)
   for y in range(mat.rows):
    for x in range(mat.cols):
      motion_stripped[y,x] = reducedColor(mat[y,x])
  else:
   for y in range(mat.rows):
    for x in range(mat.cols):
      if motion_stripped[y,x] != reducedColor(mat[y,x]):
       motion_stripped[y,x] = (0.0,0.0,0.0)
   pass
  framenum += 1
  '''
'''
for i in range(10):
  frame = cv.QueryFrame(vid)
  sub = cv.GetSubRect(frame, (0, frame.height*3/4, frame.width, frame.height/4))
  mat = cv.LoadImageM(path, cv.CV_LOAD_IMAGE_UNCHANGED)
  cv.SaveImage(str(i) + '.png', sub)
'''
"""
