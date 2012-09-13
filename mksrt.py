#!/usr/bin/python

import sys
from os import chdir
from os import getcwd
from tempfile import mkdtemp
from subprocess import check_call
from subprocess import check_output
from ocr_client import getIPAddr, getOCRText
from time import sleep
from xml.etree.ElementTree import fromstring

name = sys.argv[1]
outf = open(sys.argv[2], 'w')
origdir = getcwd()
tdir = mkdtemp()
print tdir
chdir(tdir)
check_call('cp ' + origdir + '/' + name + '*' + ' .', shell=True)
ipaddr = getIPAddr()
check_call('subp2png -n -s 0 ' + name, shell=True)
dom = fromstring(open(name + '.xml').read())
for x in dom.findall('subtitle'):
  try:
    filename = x.find('image').text
    if '.png' not in filename:
      continue
    noext = filename[:filename.rindex('.png')]
    if len(filename) <= 0 or len(noext) <= 0:
      continue
    print noext
    check_call('convert %(filename)s -alpha Off %(noext)s-conv.png' % locals(), shell=True)
    ocrtext = getOCRText(noext + '-conv.png', ipaddr)
    print >> outf, x.attrib['id']
    print >> outf, x.attrib['start'], '-->', x.attrib['stop']
    print >> outf, ocrtext
    sleep(1.0)
  except:
    continue
chdir(origdir)
check_call('rm -rf ' + tdir, shell=True)
