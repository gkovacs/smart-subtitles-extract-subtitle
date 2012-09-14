#!/usr/bin/python

import cv
from cv import *

import collections
import functools
import itertools
import json
from UnionFind import UnionFind
import random
import sys

from connected_components import *

activation_threshold = 0.5

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

def getSurrounding1(img, y, x):
  left = closestPoint(img, (y,x-1))
  right = closestPoint(img, (y,x-1))
  up = closestPoint(img, (y+1,x))
  down = closestPoint(img, (y-1,x))
  return left,right,up,down

def iterImgSurrounding1(img):
  for y,x in iterImg(img):
    mid = y,x
    left = closestPoint(img, (y,x-1))
    right = closestPoint(img, (y,x-1))
    up = closestPoint(img, (y+1,x))
    down = closestPoint(img, (y-1,x))
    yield mid,left,right,up,down

def iterImgSurrounding3(img):
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

def iterImgSurrounding4(img):
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
    left2 = closestPoint(img, (y,x-2))
    right2 = closestPoint(img, (y,x+2))
    up2 = closestPoint(img, (y+2,x))
    down2 = closestPoint(img, (y-2,x))
    yield mid,left,right,up,down,leftup,leftdown,rightup,rightdown,left2,right2,up2,down2
'''
def iterImgSurrounding4(img):
  for y,x in iterImg(img):
    mid = y,x
    coords = []
    for dy in range(-2,3):
      for dx in range(-2,3):
        if abs(dx)+abs(dy) > 3:
          continue
        coords.append(closestPoint(img, (y+dy,x+dx)))
    yield tuple([mid] + coords)
'''

def iterImgSurrounding5(img):
  for y,x in iterImg(img):
    mid = y,x
    coords = []
    for dy in range(-2,3):
      for dx in range(-2,3):
        coords.append(closestPoint(img, (y+dy,x+dx)))
    yield tuple([mid] + coords)

def iterImgSurrounding6(img):
  for y,x in iterImg(img):
    mid = y,x
    coords = []
    for dy in range(-3,4):
      for dx in range(-3,4):
        if abs(dx)+abs(dy) > 5:
          continue
        coords.append(closestPoint(img, (y+dy,x+dx)))
    yield tuple([mid] + coords)

def iterImgSurrounding7(img):
  for y,x in iterImg(img):
    mid = y,x
    coords = []
    for dy in range(-3,4):
      for dx in range(-3,4):
        coords.append(closestPoint(img, (y+dy,x+dx)))
    yield tuple([mid] + coords)

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

def getCenterHorizontal5(img):
  sub = cv.GetSubRect(img, (2*img.width/5, 0, img.width/5, img.height))
  return cv.GetMat(sub)

def extractVertical(img, vstart, vend):
  sub = cv.GetSubRect(img, (0, vstart, img.width, vend+1))
  return cv.GetMat(sub)

def extractHorizontal(img, hstart, hend):
  sub = cv.GetSubRect(img, (hstart, 0, hend, img.height))
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
    img = getCenterHorizontal5(img)
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
  for surrounding in iterImgSurrounding5(img):
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

def getVerticalActivation(extracted_color_img, harris):
  activations = [0] * extracted_color_img.height
  for y,x in iterImg(extracted_color_img):
    if sum(extracted_color_img[y,x]) > 100: # and harris[y,x] > activation_threshold:
      activations[y] += 1
  return activations

def getVerticalActivationWithHarris(extracted_color_img, harris):
  activations = [0] * extracted_color_img.height
  for y,x in iterImg(extracted_color_img):
    if sum(extracted_color_img[y,x]) > 100 and harris[y,x] > activation_threshold:
      activations[y] += 1
  return activations

def getHorizontalActivationWithHarris(extracted_color_img, harris):
  activations = [0] * extracted_color_img.width
  for y,x in iterImg(extracted_color_img):
    if sum(extracted_color_img[y,x]) > 100 and harris[y,x] > activation_threshold:
      activations[x] += 1
  return activations

# assumes centered subtitle! ie, grows left and out equally
def getHorizontalStartEnd(horizontalActivation):
  average_activation = float(sum(horizontalActivation))/len(horizontalActivation)
  center = len(horizontalActivation)/2
  beststart,bestend = (center,center)
  bestval = 0
  curval = 0
  for outward in range(len(horizontalActivation)/2):
    start,end = (center-outward,center+outward)
    curval -= average_activation*2
    curval += horizontalActivation[start]
    curval += horizontalActivation[end]
    if curval > bestval:
      bestval = curval
      beststart = start
      bestend = end
  return beststart,bestend

def getVerticalStartEnd(verticalActivation):
  average_activation = float(sum(verticalActivation))/len(verticalActivation)
  bestval = 0
  beststart,bestend = (0,0)
  for start in range(len(verticalActivation)-1):
    curval = 0
    for end in range(start, len(verticalActivation)):
      curval -= average_activation
      curval += verticalActivation[end]
      if curval > bestval:
        bestval = curval
        beststart = start
        bestend = end
  return beststart,bestend

def getVideoSubtitleVerticalStartEnd(videofile):
  counts = []
  subtitle_color = getSubtitleColor(videofile)
  best_portion = getBestPortion(videofile)
  verticalActivationTotal = None
  for idx,img in iterVideo(videofile):
    if idx % 100 != 0:
      continue
    img = getBottomQuarter(img)
    if verticalActivationTotal == None:
      verticalActivationTotal = [0] * img.height
    img = getCenterHorizontal5(img)
    extracted_color_img = extractColor(img, subtitle_color)
    harris = getHarris(img)
    verticalActivation = getVerticalActivationWithHarris(extracted_color_img, harris)
    for i,v in enumerate(verticalActivation):
      verticalActivationTotal[i] += v
  vstart,vend = getVerticalStartEnd(verticalActivationTotal)
  return vstart,vend

def blackenOutsideHorizontalRegion(origimg, hstart, hlength):
  img = CreateImage((origimg.width, origimg.height), 8, 3)
  for y,x in iterImg(origimg):
    if x >= hstart and x <= hstart+hlength:
      img[y,x] = origimg[y,x]
    else:
      img[y,x] = (0,0,0)
  return img

def imgDifference(img1, img2):
  diff = 0
  for y,x in iterImg(img1):
    if sum([abs(img1[y,x][i] - img2[y,x][i]) for i in [0,1,2]]) > 100:
      diff += 1
  return float(diff) / (img1.width * img2.height)

def haveTransition(img1, img2):
  diff = imgDifference(img1, img2)
  return diff > 0.03

'''
def getSubtitleHeight(videofile):
  num_counted = 0
  total_heights = 0
  for idx,img in iterVideo(videofile):
'''
'''
def getVerticalStartEnd(verticalActivation):
  average_activation = float(sum(verticalActivation))/len(verticalActivation)
  bestval = 0
  beststart,bestend = (0,0)
  for start in range(len(verticalActivation)-1):
    curval = 0
    for end in range(start+1, len(verticalActivation)):
      curval -= average_activation/100
      curval += verticalActivation[end]
      if curval > bestval:
        beststart,bestend = start,end
  return beststart,bestend
  #return 0,len(verticalActivation)-1
'''

'''
def getSubtitleStartEndTimes(videofile):
  best_portion = getBestPortion(videofile)
  for idx,img in iterVideo(videofile):
    print best_portion
'''

def getMetadata(vidf):
  try:
    metadata = json.load(open(vidf+'.json'))
    return metadata
  except:
    subtitle_color = getSubtitleColor(vidf)
    best_portion = getBestPortion(vidf)
    vstart,vend = getVideoSubtitleVerticalStartEnd(vidf)
    metadata = {}
    metadata['subtitle_color'] = subtitle_color
    metadata['best_portion'] = best_portion
    metadata['vstart'] = vstart
    metadata['vend'] = vend
    open(vidf+'.json', 'w').write(json.dumps(metadata))
    return metadata

def whitenAll(curseq):
  img = CreateImage((curseq[0].width, curseq[0].height), 8, 3)
  for y,x in iterImg(img):
    img[y,x] = (0,0,0)
  for nimg in curseq:
    for y,x in iterImg(nimg):
      if sum(nimg[y,x]) > 100:
        img[y,x] = (255, 255, 255)
  return img

def halfVoteImages(curseq):
  img = CreateImage((curseq[0].width, curseq[0].height), 8, 3)
  min_votes = len(curseq)/2
  if min_votes == 0:
    min_votes += 1
  for y,x in iterImg(img):
    img[y,x] = (0,0,0)
    num_votes = 0
    for nimg in curseq:
      if sum(nimg[y,x]) > 100:
        num_votes += 1
    if num_votes >= min_votes:
      img[y,x] = (255,255,255)
  return img

def averageImages(curseq):
  img = CreateImage((curseq[0].width, curseq[0].height), 8, 3)
  lencurseq = float(len(curseq))
  for y,x in iterImg(img):
    num_votes = 0
    for nimg in curseq:
      if sum(nimg[y,x]) > 100:
        num_votes += 1
    ratio = num_votes / lencurseq
    img[y,x] = (255*ratio,255*ratio,255*ratio)
  return img

def invertImage(img):
  for y,x in iterImg(img):
    img[y,x] = tuple([255-v for v in img[y,x]])

def main():
  vidf = 'video.m4v'
  if len(sys.argv) > 1:
    vidf = sys.argv[1]
  metadata = getMetadata(vidf)
  subtitle_color = metadata['subtitle_color']
  best_portion = metadata['best_portion']
  vstart = metadata['vstart']
  vend = metadata['vend']
  curImg = None
  curseq = []
  for idx,img in iterVideo(vidf):
    if idx % 10 != 0:
      continue
    img = getBottomQuarter(img)
    #extracted_color_img = extractColor(img, subtitle_color)
    extracted_color_img = extractColor(img, subtitle_color)
    #connected_components = connectedComponentLabel(extracted_color_img)
    #blacklisted_components = findComponentsSpanningOutsideRange(connected_components, vstart-5, vend+5)
    #blackenBlacklistedComponents(extracted_color_img, connected_components, blacklisted_components)
    connectedComponentOutsidePermittedRegionBlacken(extracted_color_img, vstart-5, vend+5)
    vertical_extracted_color_img = extractVertical(extracted_color_img, vstart-5, vend-vstart+5)
    harris = getHarris(vertical_extracted_color_img)
    horizontalActivation = getHorizontalActivationWithHarris(vertical_extracted_color_img, harris)
    hstart,hend = getHorizontalStartEnd(horizontalActivation)
    #nimg = extractHorizontal(vertical_extracted_color_img, hstart, hend-hstart)
    nimg = blackenOutsideHorizontalRegion(vertical_extracted_color_img, hstart-5, hend-hstart+5)
    if len(curseq) > 0 and haveTransition(curImg, nimg):
      combined_img = averageImages(curseq)
      invertImage(combined_img)
      SaveImage(str(idx)+'.png', combined_img)
      curseq = []
    curImg = nimg
    curseq.append(nimg)
  return
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
    #extracted_color_img = extractColor(img, subtitle_color)
    extracted_color_img = extractColor(img, subtitle_color)
    harris = getHarris(img)
    #verticalActivation = getVerticalActivation(extracted_color_img, harris)
    verticalActivation = getVerticalActivationWithHarris(extracted_color_img, harris)
    vstart,vend = getVerticalStartEnd(verticalActivation)
    nimg = extractVertical(extracted_color_img, vstart, vend-vstart)
    #blurred_harris = CreateImage (GetSize(harris), IPL_DEPTH_32F, 1)
    #Smooth(harris, blurred_harris, smoothtype=CV_BLUR_NO_SCALE, param1=7)
    #img_intersected = removeUnactivated(extracted_color_img, blurred_harris)
    SaveImage(str(idx)+'.png', nimg)

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

if __name__ == '__main__':
  main()
