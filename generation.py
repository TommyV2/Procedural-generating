import os
import pygame
import keyboard
import math
from random import randrange, uniform
import numpy as np
from opensimplex import OpenSimplex
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time
import operator
from scipy.interpolate import interp1d
from perlin_noise import PerlinNoise
from shapely.geometry import LineString
import cv2
import random

seed1 = randrange(1,1000)
seed2 = randrange(1,1000)
seed3 = randrange(1,1000)

noise = PerlinNoise(octaves=10, seed=1)

gen = OpenSimplex(seed1)
def noise1(nx, ny):
    # Rescale from -1.0:+1.0 to 0.0:1.0
    return gen.noise2d(nx, ny) / 2.0 + 0.5

gen2 = OpenSimplex(seed2)
def noise2(nx, ny):
    # Rescale from -1.0:+1.0 to 0.0:1.0
    return gen2.noise2d(nx, ny) / 2.0 + 0.5

shape=(800, 800) #400 - 10s 800 - 42s 1600 - 171
print("=======================================")
print("World size: ", shape[0],"x",shape[1])
#elevation_map = []
elevation_map2 = []
temperature_map = []
(world_width, world_height) = shape
islands_number = randrange(1,8)
print("Islands number: ", islands_number)
islands_number = 5
fixed_scale = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
light_fixed_scale = (1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2)
hard_fixed_scale = (1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 3)
scales =[]
scales.append(fixed_scale)
scales.append(light_fixed_scale)
scales.append(hard_fixed_scale)
i = randrange(0,3)
if i == 0:
    print("Scale: No scale")
elif i == 1:
    print("Scale: Light scale")
else:
    print("Scale: Hard scale")
scale = scales[i]
#scale = fixed_scale
islands_size = 0.2 # 0.55 0 - big 1 - small
fudge_factor=1
#elevation_level = uniform(0.01, 5.0) # used as exponent
elevation_level = 5
water_level=0.3 # 0.3 island
temperature_factor = uniform(0,0.9) #0.8 # cold - 0 hot -1

print("Temperature factor:", temperature_factor)
#img = Image.new( 'RGB', shape, "black")
img2 = Image.new( 'RGB', shape, "black")
#pixels = img.load()
pixels2 = img2.load()

#Biomes:
WATER = (24, 161, 219)
ICE = (154, 245, 234)
BEACH = (185, 196, 100)
DESERT = (161, 121, 63)
GRASS = (138, 196, 100)
ROCKS = (176, 164, 146)
FOREST = (8, 84, 26)
COLD_FOREST = (52, 99, 96)
JUNGLE = (8, 84, 23)
MOUNTAINS = (51, 38, 38)
DESERT_MOUNTAINS = (97, 62, 2)
HIGH_MOUNTAINS = (92, 90, 90)
SNOW = (237, 236, 230)    

def generate_islands(width, height, n):
    heights = []
    while n:
        add = True
        x = randrange(0+width/10,width-width/10)
        y = randrange(0+height/10,height-height/10)
        h = (x,y)
        #for hg in heights:
        #    tmp_dist = math.sqrt( (x-hg[0])**2+(y-hg[1])**2)
        #    if tmp_dist < 2*width/10:
        #        add = False
        #if add == True:
        #    heights.append(h)
        #    n-=1
        heights.append(h)
        n-=1
    return heights

def islands_with_eye_function(x, y, heights, w):
    d1 = w/4
    d2 = w/5
    inc = 0
    dist = 0
    n = len(heights)
    for height in heights:
        height_x = height[0]
        height_y = height[1]
        tmp_dist = math.sqrt( (x-height_x)**2+(y-height_y)**2)
    if tmp_dist <= d1 and tmp_dist >= d2:
        dist = 0.1
    elif tmp_dist > d1:
        dist = 0.1+4*(tmp_dist-d1)/w
    else:
        dist = 0.1+4*(d2-tmp_dist)/w #d2 mniejsze
    if n > 1:
        dist = 0.1+0.3*n*dist # 0.55 0 - big 1 - small islands_size
    
    return dist

def islands_function(x, y, heights, w, scale):
    dist = 10000
    n = len(heights)
    for height in heights:
        height_x = height[0]
        height_y = height[1]
        tmp_dist = math.sqrt( (x-height_x)**2+(y-height_y)**2)
        if tmp_dist <= dist:
            dist = tmp_dist    
    if n > 10:
        dist = 3*dist # 0.55 0 - big 1 - small islands_size
    else:
        dist = scale[n-1]*dist
    return dist*2/w

def get_biome(e, m):    
    if e<water_level:
        if m > temperature_factor+0.4 and temperature_factor<0.1:
            return ICE
        return WATER

    if e<water_level+0.1:
        if m>temperature_factor+0.4 or temperature_factor<0.1:
            return ROCKS
        return BEACH

    if e>water_level+0.7:
        if m < temperature_factor:
            return DESERT_MOUNTAINS
        if m < temperature_factor+ 0.2:
            return MOUNTAINS
        if m < temperature_factor+0.4 or temperature_factor>0.1:
            return HIGH_MOUNTAINS
        return SNOW

    if e>water_level+0.5:
        if m < temperature_factor+0.13:
            return DESERT
        if m < temperature_factor+0.44 or temperature_factor>0.1: #0.44
            return GRASS
        return SNOW

    if e>water_level+0.2:
        if m < temperature_factor:
            return DESERT
        if m < temperature_factor+0.2:
            return JUNGLE
        if m < temperature_factor+0.4 or temperature_factor>0.1:
            return FOREST
        return COLD_FOREST

    if m < temperature_factor:
        return DESERT
    if m < temperature_factor+0.13:
        return GRASS
    if m < temperature_factor+0.4 or temperature_factor>0.1:
        return ROCKS
    return SNOW

def get_biome2(e, m):    
    if e<water_level:
        if m > temperature_factor+0.4:
            return ICE
        return WATER

    if e<water_level+0.1:
        if m>temperature_factor+0.4:
            return ROCKS
        return BEACH

    if e>water_level+0.7:
        if m < temperature_factor:
            return DESERT_MOUNTAINS
        if m < temperature_factor+ 0.2:
            return MOUNTAINS
        if m < temperature_factor+0.4:
            return HIGH_MOUNTAINS
        return SNOW

    if e>water_level+0.5:
        if m < temperature_factor+0.13:
            return DESERT
        if m < temperature_factor+0.44: #0.44
            return GRASS
        return SNOW

    if e>water_level+0.2:
        if m < temperature_factor:
            return DESERT
        if m < temperature_factor+0.2:
            return JUNGLE
        if m < temperature_factor+0.4:
            return FOREST
        return COLD_FOREST

    if m < temperature_factor:
        return DESERT
    if m < temperature_factor+0.13:
        return GRASS
    if m < temperature_factor+0.4:
        return ROCKS
    return SNOW

def thermal_erosion(elevation):
    T = 0
    c = 1
    eroded_map = elevation
    width = shape[0]-1
    height = shape[1]-1
    for y in range(1, height):
        for x in range(1, width):
            y1 = y - 1
            y2 = y + 1
            x1 = x - 1
            x2 = x + 1   

            di = eroded_map[x][y] - eroded_map[x1][y1] 
            if di > T:         
                eroded_map[x1][y1] += c*(di - T)
                eroded_map[x][y] -= c*(di - T) 

            di = eroded_map[x][y] - eroded_map[x][y1] 
            if di > T:            
                eroded_map[x][y1] += c*(di - T) 
                eroded_map[x][y] -= c*(di - T) 

            di = eroded_map[x][y]-eroded_map[x2][y1] 
            if di > T:
                eroded_map[x2][y1] += c*(di - T)
                eroded_map[x][y] -= c*(di - T)  

            di = eroded_map[x][y]-eroded_map[x][y1] 
            if di > T:
                eroded_map[x1][y] += c*(di - T) 
                eroded_map[x][y] -= c*(di - T)

            di = eroded_map[x][y]-eroded_map[x2][y] 
            if di > T:
                eroded_map[x2][y] += c*(di - T) 
                eroded_map[x][y] -= c*(di - T)

            di = eroded_map[x][y]-eroded_map[x1][y2] 
            if di > T:
                eroded_map[x1][y2] += c*(di - T)
                eroded_map[x][y] -= c*(di - T)

            di = eroded_map[x][y]-eroded_map[x][y2] 
            if di > T:
                eroded_map[x][y2] += c*(di - T)
                eroded_map[x][y] -= c*(di - T)

            di = eroded_map[x][y]- eroded_map[x2][y2] 
            if di > T:
                eroded_map[x2][y2] += c*(di - T)
                eroded_map[x][y] -= c*(di - T)  
    return eroded_map

def hydraulic_erosion(temperature, rivers):
    scale = 10
    eroded_map = temperature
    width = shape[0]-1
    height = shape[1]-1
    for y in range(1, height):
        for x in range(1, width): 
            done = False       
            for river in rivers:
                if done == True:
                    break
                for point in river:
                    river_x = point[0]
                    river_y = point[1]
                    tmp_dist = math.sqrt( (x-river_x)**2+(y-river_y)**2)
                    if tmp_dist <= 4: #4
                        eroded_map[y][x] -= (8-tmp_dist)/15
                        done = True
                        break               
    return temperature

def temperature_erosion(elevation, riverss):
    scale = 10
    eroded_map = elevation
    width = shape[0]-1
    height = shape[1]-1
    for y in range(1, height):
        for x in range(1, width): 
            done = False       
            for river in riverss:
                if done == True:
                    break
                for point in river:
                    river_x = point[0]
                    river_y = point[1]
                    tmp_dist = math.sqrt( (x-river_x)**2+(y-river_y)**2)
                    if tmp_dist <= 8: #4
                        eroded_map[y][x] += (8-tmp_dist/16)/16 
                        done = True
                        break           
            
        
    return eroded_map

start = time.time()

def is_inside_river(rivers, point, radius):
    for river in rivers:
        for river_segment in river:
            d = math.sqrt( ((river_segment[0]-point[0])**2)+((river_segment[1]-point[1])**2))
            if d <= radius:
                return True
    return False

def is_big_enought(elevation_map, angles, p):
    px = p[0]
    py = p[1]
    count = 0
    for angle in angles:            
        x = px + math.cos(angle)*10
        y = py + math.sin(angle)*10
        if x > world_width or y > world_height or x <0 or y<0:
            continue
        if elevation_map[int(y)][int(x)] < water_level:
            count +=1
    if count >=4:
        return True
    return False

heights = generate_islands(shape[0], shape[1], islands_number)
#heights = [(300,500),(400, 500),(500,500), (400, 600), (400, 700)]
for y in range(world_height):
    #elevation_map.append([0] * world_width)
    elevation_map2.append([0] * world_width)
    temperature_map.append([0] * world_width)
    for x in range(world_width):       
        nx=x/world_width - 0.5
        ny=y/world_height - 0.5
        
    
    #    d1 = 2 * max(abs(nx), abs(ny)) 
    #    d2 = math.sqrt(nx*nx + ny*ny) / math.sqrt(0.3) # wyspa 
    #    print("nx: ", nx, " ny: ", ny, " d2: ",d2)
    #    d3 = abs(nx) + abs(ny)
    #    d4 = math.sqrt(nx*nx + ny*ny) / math.sqrt(4) # pasmo gorskie
    #    d5 = 1-(math.sqrt(nx*nx + ny*ny) / math.sqrt(0.3)) # zatoka
    #    d6 = 3*abs(ny)  # pasmo poziome
    #    d7 = 2*abs(nx)  # pasmo pionowe       
    #    d8 = 0.4 # 0- góry 1 - woda random
    #    t1 = 1-3*abs(ny) # temperatura na biegunach
    #    custom = islands_function(y, x, heights, shape[0])
        custom = islands_function(y, x, heights, shape[0], scale)
        e = (0.54 * noise1( 4 * nx,  4 * ny)
           + 0.50 * noise1( 8 * nx,  8 * ny)
           + 0.25 * noise1( 16 * nx,  16 * ny)
           + 0.13 * noise1( 64 * nx,  64 * ny))
          # + 0.06 * noise1(64 * nx, 64 * ny)
          # + 0.03 * noise1(64 * nx, 64 * ny)
          # + 0.015 * noise1(128 * nx, 128 * ny))
        #print("d2: ",d2," c: ",custom)
        e = 1+e-custom# 1 shape
        #e = d2 +e*(1-d2)
        e = e / (0.54 + 0.50 + 0.25+ 0.13 + 0.06+ 0.03+0.015) #     
        e = (e*fudge_factor)**(elevation_level)       
        
        #e = round(e*32)/32
        #e = (e) / (0.54 + 0.50 + 0.25 + 0.13 + 0.06 + 0.03)
        
        m = (1.00 * noise2( 1 * nx,  1 * ny)
           + 0.75 * noise2( 2 * nx,  2 * ny)
           + 0.33 * noise2( 4 * nx,  4 * ny)
           + 0.33 * noise2( 8 * nx,  8 * ny)
           + 0.33 * noise2(16 * nx, 16 * ny)
           + 0.50 * noise2(32 * nx, 32 * ny))           
        m = (m) / (1.00 + 0.75 + 0.33 + 0.33 + 0.33+0.50)    
        #elevation_map[y][x] = e
        elevation_map2[y][x] = e
        temperature_map[y][x] = m
        #biome=get_biome(e, m) 
        #pixels[x,y] = biome
'''
elevation_map2 = thermal_erosion(elevation_map2)
elevation_map2 = hydraulic_erosion(elevation_map2, rivers):
if elevation_map2 == elevation_map:
    print("Same")
else:
    print("Different")
for y in range(world_height):
    for x in range(world_width):
        e = elevation_map[y][x]
        e2 = elevation_map2[y][x]
        m = temperature_map[y][x]
        biome=get_biome(e, m)
        biome2= get_biome(e2, m)
        pixels[x,y] = biome
        pixels2[x,y] = biome2
'''
#rivers
rivers_number = randrange(8,15)
print("Rivers number: ",rivers_number)
i=0 
starting_points = []
ending_points = []
tries = 0

while i < rivers_number:
    x = randrange(10,world_width-10)
    y = randrange(10,world_height-10)
    if elevation_map2[x][y] > water_level+0.7: 
        point= (y,x)
        starting_points.append(point)
        i+=1
       
    
angles = np.arange(0.0, 371.25, 11.25)
angles2 = np.arange(0.0, 405, 55)
def generate_river(rivers, starting_point, elevation_map, angles, is_extra_river):
    line_len = 3
    river = [] 
    
    current_x = int(starting_point[0])
    current_y = int(starting_point[1])
    current_height = elevation_map[current_y ][current_x] 
    p = (current_x, current_y)
    river.append(p)
    segments = 0
    while True:
        start_angle = 0
        steepness = 0
        tmp_x = 0
        tmp_y = 0
        tmp_height = 0
        tmp_list = []
        max_steepness = 0
        i = 0
        loop = True
        end = False
        while loop:
            for angle in angles:            
                x = current_x + math.cos(angle)*(line_len+i*2)
                y = current_y + math.sin(angle)*(line_len+i*2)
                if x >= shape[0] or y >= shape[1] or x <= 0 or y <= 0:
                    tmp_x = current_x       
                    loop = False
                    end = True
                    tmp_y = current_y
                    tmp_height = current_height
                    break
                height = elevation_map[int(y)][int(x)]
                if is_extra_river:
                    new_steepness = current_height - height
                else:
                    new_steepness = height - current_height
                tmp_list.append(new_steepness)
                if new_steepness < steepness:
                    tmp_x = x
                    loop = False
                    tmp_y = y
                    tmp_height = height
                    steepness = new_steepness
                
            i=i+1
        current_height = tmp_height       
        current_x = tmp_x
        current_y = tmp_y
        p = (current_x, current_y)
        if is_extra_river == True:
            if is_inside_river(rivers, p, 3) == True:
                break   
        river.append(p)
        if len(river)>1:
            previous = river[-2]
            d = math.sqrt( ((p[0]-previous[0])**2)+((p[1]-previous[1])**2))
            if d > 30:
                new = get_smaller_river_chunks(previous, p)
                river.pop()
                river.extend(new)
        segments +=1
        if (is_extra_river and segments >= 14):
            end = True  
        if end == True:
            break
        if current_height < water_level:
            if is_big_enought(elevation_map, angles2, p) == True:
                break
        #if (current_height < water_level and is_big_enought(elevation_map, angles2, p)) or end == True:
        #    break
    return river 

def get_smaller_river_chunks(start, end): # zamiast end mamy liste punktów
    lineLen = 5            
    maxAngle = math.radians(90)        
    minDistToEnd = 8 
    current = start
    new_points = []
    while True:
        xDist = end[0] - current[0]
        yDist = end[1] - current[1]
        between = math.atan2(yDist, xDist)   
        newAngle = between + (uniform(0,1) * maxAngle - maxAngle/2)
        x = current[0] + math.cos(newAngle) * lineLen
        y = current[1] + math.sin(newAngle) * lineLen
        point = (x,y)
        new_points.append(point)
        current = point
        distLeft = math.sqrt( ((end[0]-current[0])**2)+((end[1]-current[1])**2))

        if distLeft <= minDistToEnd:
            new_points.append(end)
            break
    return new_points

TEST_RED = (204, 0, 0)
#draw = ImageDraw.Draw(img)
draw2 = ImageDraw.Draw(img2)
m = 0
rivers = []
extra_rivers = []
for point in starting_points:
    print(m)
    extra = randrange(8, 15)
    river = generate_river(rivers, point, elevation_map2, angles, False)
    if extra > len(river):
        extra = len(river)
    extra_points = river[-extra:] #random.sample(river, extra)
    rivers.append(river)
    for extra_point in extra_points:
        extra_river = generate_river(extra_rivers, extra_point, elevation_map2, angles, True)
        extra_rivers.append(extra_river)
    #draw.line(river, fill=WATER, width=4)
    #draw2.line(river, fill=WATER, width=4)
    
    m+=1

elevation_map2 = hydraulic_erosion(elevation_map2, rivers)
#elevation_map2 = hydraulic_erosion(elevation_map2, extra_rivers)

elevation_map2 = thermal_erosion(elevation_map2)
temperature_map2 = temperature_map
if temperature_factor > 0.4:
    temperature_map2 = temperature_erosion(temperature_map, rivers)

for y in range(world_height):
    for x in range(world_width):
        #e = elevation_map[y][x]
        e2 = elevation_map2[y][x]
        #m = temperature_map[y][x]
        m2 = temperature_map2[y][x]
        #biome=get_biome(e, m)
        biome2= get_biome(e2, m2)
        '''
        if temperature_factor > 0.4:
            biome2= get_biome2(e2, m2)
        else:
            biome2= get_biome(e2, m2)
        '''
        #pixels[x,y] = biome
        pixels2[x,y] = biome2

for river in rivers:
    #river = generate_river(point, elevation_map, angles)
    #draw.line(river, fill=WATER, width=6)
    draw2.line(river, fill=WATER, width=4)
    m+=1
#print("Extra rivers:",len(extra_rivers))
#print(extra_rivers)
for extra_river in extra_rivers:
    #river = generate_river(point, elevation_map, angles)
    #draw.line(extra_river, fill=WATER, width=3)
    draw2.line(extra_river, fill=WATER, width=3)
    #m+=1


end = time.time()
print("Time: ", end-start)
print("=======================================")

#img.show()
img2.show()
#img = img.save("map1.jpg") 
img2 = img2.save("map1_erosion.jpg") 

'''
plt.figure()
pic2 = [[islands_function(i, j, heights, shape[0], scale) for j in range(shape[0])] for i in range(shape[1])]
pic = [[elevation_map[i][j] for j in range(shape[0])] for i in range(shape[1])]
pic3 = [[elevation_map2[i][j] for j in range(shape[0])] for i in range(shape[1])]

plt.imshow(pic, cmap='gray')
plt.show()
plt.imshow(pic2, cmap='gray')
plt.show()
plt.imshow(pic3, cmap='gray')
plt.show()
'''
