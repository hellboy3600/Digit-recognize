import sys
import cv2
import tensorflow as tf
from PIL import Image, ImageFilter
import numpy
import PIL.ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import imutils
import math

def predictint():

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
       
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, r"model2.ckpt")
        prediction=tf.argmax(y_conv,1)
        im = cv2.imread(img_name)
        im = cv2.bitwise_not(im)
        print(img_name)
        height, width = im.shape[:2]
        x1 = round(width*70/1654)
        y1 = round(height*25/2339)
        height, width = im.shape[:2]
        print(x1,y1)
	    
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,255,255,255)
        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        maxsize = 0
        count = 0
        max = 0
        a=[]
        for c1 in cnts:
         # compute the center of the contour 
        
            M = cv2.moments(c1)
            x2,y2,w2,h2 = cv2.boundingRect(c1)
            if (h2>x1-10 and w2>x1*2-20 and w2<x1*2+20):
                print("MATCH")
            if (h2<x1+30 and h2>x1-10 and ((w2>x1*2-20 and w2<x1*2+20) or (w2>x1*5-10 and w2<x1*5+10))):
                count = count+1
                l = int(w2/x1)
                c = 0
                a=[]
                while (c<l):
                 cr = image[y2+5:y2+h2-5,x2+x1*(c)+5:x2+x1*(c+1)]

                 c = c+1
                 cv2.imwrite('periodic.png',cr)
                 flag = 0
                 flag = conv('periodic.png')
                 
                 imvalue = imageprepare()
                 if flag==1:
                  s = prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)
                  a.append(s[0])
                 else:
                  s = "None"
                  a.append(s)
                print(a)
                cropped = image[y2+5:y2+h2-5,x2+5:x2+w2-5]
                cv2.imshow("Image", cropped)
                cv2.waitKey(0)
      

        return(a)
		
def conv(img):

    image = cv2.imread(img)
    rows,cols = image.shape[:2]

    img = cv2.GaussianBlur(image,(5,5),0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh1 = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
     
    best = 0
    maxsize = 0
    count = 0
    flag = 0
    for c in cnts:
     # compute the center of the contour
        M = cv2.moments(c)
        flag = 1
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if cv2.contourArea(c) > maxsize:
            maxsize = cv2.contourArea(c)
            best = count

    height, width = image.shape[:2]
    if (flag==1):
		[vx,vy,x,y] = cv2.fitLine(cnts[best], cv2.DIST_L2,0,0.01,0.01)
	
		x,y,w,h = cv2.boundingRect(cnts[best])
		x3,y3,w3,h3 = cv2.boundingRect(cnts[best])
		hh = int(height/2-h/2)
		hh1 = int(height/2+h/2)
		ww = int(width/2-w/2)
		ww1 = int(width/2+w/2)
		cropped=img[y:y+h,x:x+w]

		cv2.waitKey(0)
		if (w>h-10):
		cropped=cv2.resize(cropped,(25,30))
		gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (1, 1), 0)
		thresh1 = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)[1]

		cnts = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_NONE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		best = 0
		maxsize = 0
		count = 0
		flag = 0
		for c in cnts:

			M = cv2.moments(c)
			flag = 1
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		
			if cv2.contourArea(c) > maxsize:
				maxsize = cv2.contourArea(c)
				best = count

		cv2.drawContours(cropped, [cnts[best]], -1, (0,0,0), 1)
		x,y,w,h = cv2.boundingRect(cnts[best])
		x3,y3,w3,h3 = cv2.boundingRect(cnts[best])
		hh = int(height/2-h/2)
		hh1 = int(height/2+h/2)
		ww = int(width/2-w/2)
		ww1 = int(width/2+w/2)
		[vx,vy,x,y] = cv2.fitLine(cnts[best], cv2.DIST_L2,0,0.01,0.01)

		cropped = cv2.bitwise_not(cropped)
		cv2.imwrite('sample.png',cropped)
		
		cv2.imshow("Cropped",cropped)
		cv2.imwrite('sample.png',cropped)
		
		cv2.imshow("Cropped",cropped)
		cv2.waitKey(0)
		tva = imageprepare()     
		return(1)
    else:
     image = cv2.imread('zero.png')
     cv2.imwrite('sample.png',image)
     tva = imageprepare()
    return(0)
    
def imageprepare():

    newImage = Image.new('L', (28, 28), (255))
    img = Image.open("sample.png")
    im = img.convert('L')
    width = float(im.size[0])
    height = float(im.size[1])

    if width > 500+height: 
        nheight = int(round((20.0/width*height),0)) 
        if (nheigth == 0): 
            nheigth = 1  

        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0))
        newImage.paste(img, (0, wtop)) 

    else:

        nwidth = int(round((20.0/height*width),0)) 
        if (nwidth == 0): 
            nwidth = 1

        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        img.save("sample.png")
        wleft = int(round(((28 - nwidth)/2),0)) 
        newImage.paste(img, (wleft,4)) 
        
    
    tv = list(newImage.getdata())
    
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva

def main(argv):

    global img_name
    img_name = str(argv)
    predint = predictint()
    
if __name__ == "__main__":
    main(sys.argv[1])