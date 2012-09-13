#!/usr/bin/python

# Sends png file to OneNote OCR service and prints output.
#
# Copyright Geza Kovacs <gkovacs@mit.edu>

import urllib2
import httplib
import base64
import sys

def getIPAddr():
  #return urllib2.urlopen('http://transgame.csail.mit.edu:9537/?varname=win7ipnetbook').read()
  return urllib2.urlopen('http://transgame.csail.mit.edu:9537/?varname=win7ipaddress').read()

def getOCRText(png_file_to_ocr, ipaddr):
  httpServ = httplib.HTTPConnection(ipaddr, 8080)
  httpServ.connect()

  data = base64.b64encode(open(png_file_to_ocr).read())

  httpServ.request('POST', '/', data)

  response = httpServ.getresponse()
  retv = ""
  if response.status == httplib.OK:
    retv = response.read()
    print retv
  else:
    print "Got error from server:", response.status
  httpServ.close()
  return retv

def main():
  png_file_to_ocr = sys.argv[1]
  print png_file_to_ocr
  ipaddr = getIPAddr()
  print ipaddr
  print getOCRText(png_file_to_ocr, ipaddr)

if __name__ == "__main__":
  main()
