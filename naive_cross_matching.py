# Write your hms2dec and dms2dec functions here
def hms2dec(hour,minute,second):
  result = 15*(hour + (minute/60) + (second/3600))
  return result
def dms2dec(degree,minute,second):
  if degree<0:
    sign = -1
  else:
    sign = 1
  return sign*(abs(degree) + (minute/60) + (second/3600))

# Write your angular_dist function here using HAVERSINE FORMULA
import numpy as np
def angular_dist(ra1,dec1,ra2,dec2):
  ra1_rad = np.radians(ra1)
  dec1_rad = np.radians(dec1)
  ra2_rad = np.radians(ra2)
  dec2_rad = np.radians(dec2)
  a = np.sin(np.abs(dec1_rad - dec2_rad)/2)**2
  b = np.cos(dec1_rad)*np.cos(dec2_rad)*np.sin(np.abs(ra1_rad - ra2_rad)/2)**2
  result = 2*np.arcsin(np.sqrt(a+b))
  return np.degrees(result)
  
 def import_bss():
  res = []
  cat = np.loadtxt('bss.dat', usecols=range(1,7))
  for i,row in enumerate(cat,1):
    res.append((i,hms2dec(row[0], row[1], row[2]), dms2dec(row[3], row[4], row[5])))
  return res
 def import_super():
  cat = np.loadtxt('super.csv',delimiter=',', skiprows=1, usecols=(0,1))
  res = []
  for i,row in enumerate(cat,1):
    res.append((i, row[0], row[1]))
  return res
  
  def find_closest(cat, ra, dec):
  min_distance = np.inf
  min_id = None
  for id1, ra1, dec1 in cat:
    distance = angular_dist(ra1,dec1,ra,dec)
    if distance < min_distance:
      min_id = id1
      min_distance = distance
  return min_id,min_distance

def crossmatch(cat1,cat2,max_radius):
  matches = []
  no_matches = []
  for id1, ra1, dec1 in cat1:
    closest_distance = np.inf
    closest_id2 = None
    for id2, ra2, dec2 in cat2:
      dist = angular_dist(ra1, dec1, ra2, dec2)
      if dist < closest_distance:
        closest_id2 = id2
        closest_distance = dist
    if closest_distance > max_radius:
      no_matches.append(id1)
    else:
      matches.append((id1,closest_id2,closest_distance))
  return matches, no_matches
 


%%%%%%%%%%% main function to calculate cross matching %%%%%%%%
# You can use this to test your function.
# Any code inside this `if` statement will be ignored by the automarker.
  bss_cat = import_bss()
  super_cat = import_super()

  # First example in the question
  max_dist = 40/3600
  matches, no_matches = crossmatch(bss_cat, super_cat, max_dist)
  print(matches[:3])
  print(no_matches[:3])
  print(len(no_matches))

%%%%%%%%%%% main function to calculate closest distance %%%%%%%%
  cat = import_bss()
  
  # First example from the question
  print(find_closest(cat, 175.3, -32.5))

  # Second example in the question
  print(find_closest(cat, 32.2, 40.7))




