import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def change_brightness(image, value):
    img = image.copy()  
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(3):
                if img[i,j,k] + value > 255:
                    img[i,j,k] = 255
               
                elif img[i,j,k] + value < 0:
                    img[i,j,k] = 0
                
                else:
                    img[i,j,k] += value
    return img

def change_contrast(image, value):
    img = image.copy()  
    F = (259*(value + 255))/(255*(259-value))
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(3):
               C = img[i,j,k]
               if F*(C-128) + 128 > 255:
                   img[i,j,k] = 255
               elif F*(C-128) + 128 < 0:
                   img[i,j,k] = 0
               else:
                   img[i,j,k] = F*(C-128) + 128   
    return img
        
def grayscale(image):    
    img = image.copy()  
    for i in range(len(img)):
        for j in range(len(img[0])):
            R = img[i,j,0]
            G = img[i,j,1]
            B = img[i,j,2]
            new = 0.3*R + 0.59*G + 0.11*B
            for k in range(3):
                img[i,j,k] = new
               
    return img
              
def blur_effect(image):
    img = image.copy()      
    r = len(img)
    c = len(img[0])    
    img_new = np.full((r, c, 3), 0)

    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            for k in range(3):
                tl = img[i-1,j-1,k]
                t  = img[i-1,j  ,k]
                tr = img[i-1,j+1,k]
                l  = img[i  ,j-1,k]
                p  = img[i  ,j  ,k]
                r  = img[i  ,j+1,k]
                bl = img[i+1,j-1,k]
                b  = img[i+1,j  ,k]
                br = img[i+1,j+1,k]
                img_new[i,j,k] = 0.0625*tl + 0.125*t + 0.0625*tr + 0.125*l + 0.25*p + 0.125*r + 0.0625*bl + 0.125*b + 0.0625*br
                
    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            for k in range(3):
                img[i,j,k] = img_new[i,j,k]
    return img
        
def edge_detection(image):
    img = image.copy()    
    r = len(img)
    c = len(img[0])   
    img_new = np.full((r, c, 3), 0)
    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            for k in range(3):               
                tl = img[i-1,j-1,k]
                t  = img[i-1,j  ,k]
                tr = img[i-1,j+1,k]
                l  = img[i  ,j-1,k]
                p  = img[i  ,j  ,k]
                r  = img[i  ,j+1,k]
                bl = img[i+1,j-1,k]
                b  = img[i+1,j  ,k]
                br = img[i+1,j+1,k]
                img_new[i,j,k] = -1*tl + -1*t + -1*tr + -1*l + 8*p + -1*r + -1*bl + -1*b + -1*br + 128
                
                if img_new[i,j,k] < 0:
                    img_new[i,j,k] = 0
                elif img_new[i,j,k] > 255:
                    img_new[i,j,k] = 255
                                    
    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            for k in range(3):
                img[i,j,k] = img_new[i,j,k]
    return img

def embossed(image):
    img = image.copy()    
    r = len(img)
    c = len(img[0])  
    img_new = np.full((r, c, 3), 0)

    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            for k in range(3):              
                tl = img[i-1,j-1,k]
                t  = img[i-1,j  ,k]
                tr = img[i-1,j+1,k]
                l  = img[i  ,j-1,k]
                p  = img[i  ,j  ,k]
                r  = img[i  ,j+1,k]
                bl = img[i+1,j-1,k]
                b  = img[i+1,j  ,k]
                br = img[i+1,j+1,k]
                img_new[i,j,k] = -1*tl + -1*t + 0*tr + -1*l + 0*p + 1*r + 0*bl + 1*b + 1*br + 128
                
                if img_new[i,j,k] < 0:
                    img_new[i,j,k] = 0
                elif img_new[i,j,k] > 255:
                    img_new[i,j,k] = 255
                                     
    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            for k in range(3):
                img[i,j,k] = img_new[i,j,k]
    return img

def rectangle_select(image, x, y):  
    mask = np.zeros((len(image),len(image[0])))   
    for i in range(x[0],y[0]+1):
        for j in range(x[1],y[1]+1):  
            mask[i][j] = 1          
    return mask
    
def magic_wand_select(image, x, thres):                   
    row = len(image)
    col = len(image[0])
    mask = np.zeros((len(image),len(image[0])))
    mask[x[0]][x[1]] = 1
    
    stack = [x]
    done = [x]
        
    while (len(stack) != 0):
        
        current = stack.pop()        
        cr = image[x[0]][x[1]][0]
        cg = image[x[0]][x[1]][1]
        cb = image[x[0]][x[1]][2]
        
        top = (current[0]-1,current[1])              
        if top[0] >= 0 and top[0] < row and top[1] >= 0 and top[1] < col and top not in done:       
            
            tr = image[top[0]][top[1]][0]
            tg = image[top[0]][top[1]][1]
            tb = image[top[0]][top[1]][2]            
            r = (cr + tr)/2           
            
            dist = (2+ r/256) * pow((tr-cr),2) + 4 * pow((tg-cg),2) + (2 + (255-r)/256) * pow((tb-cb),2)
            dist = pow(dist, 0.5)
                        
            if dist <= thres:
                mask[top[0]][top[1]] = 1
                stack.append(top)
            done.append(top)
            
        bot = (current[0]+1,current[1])            
        if bot[0] >= 0 and bot[0] < row and bot[1] >= 0 and bot[1] < col and bot not in done:
            
            br = image[bot[0]][bot[1]][0]
            bg = image[bot[0]][bot[1]][1]
            bb = image[bot[0]][bot[1]][2]           
            r = (cr + br)/2
            
            dist = (2 + r/256) * pow((br-cr),2) + 4 * pow((bg-cg),2) + (2 + (255-r)/256) * pow((bb-cb),2)
            dist = pow(dist, 0.5)

            if dist <= thres:
                mask[bot[0]][bot[1]] = 1
                stack.append(bot)
            done.append(bot)
            
        lf = (current[0],current[1]-1)              
        if lf[0] >= 0 and lf[0] < row and lf[1] >= 0 and lf[1] < col and lf not in done:
            
            lr = image[lf[0]][lf[1]][0]
            lg = image[lf[0]][lf[1]][1]
            lb = image[lf[0]][lf[1]][2]
        
            r = (cr + lr)/2
            dist = (2+ r/256) * pow((lr-cr),2) + 4 * pow((lg-cg),2) + (2 + (255-r)/256) * pow((lb-cb),2)
            dist = pow(dist, 0.5)

            if dist <= thres:
                mask[lf[0]][lf[1]] = 1
                stack.append(lf)
            done.append(lf)
            
        rh = (current[0],current[1]+1)              
        if rh[0] >= 0 and rh[0] < row and rh[1] >= 0 and rh[1] < col and rh not in done:
            
            rr = image[rh[0]][rh[1]][0]
            rg = image[rh[0]][rh[1]][1]
            rb = image[rh[0]][rh[1]][2]
            r = (cr + rr)/2
            
            dist = (2+ r/256) * pow((rr-cr),2) + 4 * pow((rg-cg),2) + (2 + (255-r)/256) * pow((rb-cb),2)
            dist = pow(dist, 0.5)

            if dist <= thres:
                mask[rh[0]][rh[1]] = 1
                stack.append(rh)
            done.append(rh)

    return mask

def compute_edge(mask):           
    rsize, csize = len(mask), len(mask[0]) 
    edge = np.zeros((rsize,csize))
    if np.all((mask == 1)): return edge        
    for r in range(rsize):
        for c in range(csize):
            if mask[r][c]!=0:
                if r==0 or c==0 or r==len(mask)-1 or c==len(mask[0])-1:
                    edge[r][c]=1
                    continue
                
                is_edge = False                
                for var in [(-1,0),(0,-1),(0,1),(1,0)]:
                    r_temp = r+var[0]
                    c_temp = c+var[1]
                    if 0<=r_temp<rsize and 0<=c_temp<csize:
                        if mask[r_temp][c_temp] == 0:
                            is_edge = True
                            break
    
                if is_edge == True:
                    edge[r][c]=1
            
    return edge

def save_image(filename, image):
    img = image.astype(np.uint8)
    mpimg.imsave(filename,img)


def load_image(filename):
    img = mpimg.imread(filename)
    if len(img[0][0])==4: # if png file
        img = np.delete(img, 3, 2)
    if type(img[0][0][0])==np.float32:  # if stored as float in [0,..,1] instead of integers in [0,..,255]
        img = img*255
        img = img.astype(np.uint8)
    mask = np.ones((len(img),len(img[0]))) # create a mask full of "1" of the same size of the laoded image
    img = img.astype(np.int32)
    return img, mask

def display_image(image, mask):
    # if using Spyder, please go to "Tools -> Preferences -> IPython console -> Graphics -> Graphics Backend" and select "inline"
    tmp_img = image.copy()
    edge = compute_edge(mask)
    for r in range(len(image)):
        for c in range(len(image[0])):
            if edge[r][c] == 1:
                tmp_img[r][c][0]=255
                tmp_img[r][c][1]=0
                tmp_img[r][c][2]=0
 
    plt.imshow(tmp_img)
    plt.axis('off')
    plt.show()
    print("Image size is",str(len(image)),"x",str(len(image[0])))

def menu():
    
    img = mask = np.array([])  
   
   
            
    
    while (len(img) == 0):
        user_input = input("What do you want to do ?\ne - exit\nl - load a picture\n")
        if user_input == "e":
            return 0
        elif user_input == "l":
            filename = input("Enter filename: ")
            img, mask = load_image(filename)
        else:
            user_input = input("What do you want to do ?\ne - exit\nl - load a picture\n")
    
    while (user_input != "e"):
        
        
        img2 = np.full((len(img), len(img[0]), 3), 0)
    
        for r in range(len(img)):
            for c in range(len(img[0])):
                for k in range(3):
                    img2[r][c][k] = img[r][c][k]
        
                                             
        user_input = input("e - exit\nl - load a picture\ns - save the current picture\n1 - adjust brightness\n2 - adjust contrast\n3 - apply grayscale\n4 - apply blur\n5 - edge detection\n6 - embossed\n7 - rectangle select\n8 - magic wand select\n")
        
        if user_input == "l":
            filename = input("Enter filename: ")
            img, mask = load_image(filename)
            
        elif user_input == "s":
            save_image(filename, img)
        
        elif user_input == "1":
            value = input("Enter Value: ")
            is_int = False
            while (is_int == False):
                try:
                    # convert to integer
                    int(value)
                    
                except ValueError:
                    is_int = False
                    value = input("Error! Enter Value: ")
                    
                else:
                    if (int(value)) >= -255 and (int(value)) <= 255:
                        is_int = True
                        value = int(value)
                    else:
                        value = input("Error! Enter Value: ")
                        
                    
            img = change_brightness(img, value)
            
        elif user_input == "2":
            value = input("Enter Value: ")
            is_int = False
            while (is_int == False):
                try:
                    # convert to integer
                    int(value)
                    
                except ValueError:
                    is_int = False
                    value = input("Error! Enter Value: ")
                    
                else:
                    if (int(value)) >= -255 and (int(value)) <= 255:
                        is_int = True
                        value = int(value)
                    else:
                        value = input("Error! Enter Value: ")
            
            img = change_contrast(img, value)
            
        elif user_input == "3":
            img = grayscale(img)
            
        elif user_input == "4":
            img = blur_effect(img)
            
        elif user_input == "5":
            img = edge_detection(img)
            
        elif user_input == "6":
            img = embossed(img)
            
        elif user_input == "7":
            is_rectangle = False
            while is_rectangle == False:
                
                r = len(img)
                c = len(img[0])
                
                value = input("Enter top left pixel row: ")
                is_int = False
                while (is_int == False):
                    try:
                        # convert to integer
                        int(value)
                        
                    except ValueError:
                        is_int = False
                        value = input("Error! Enter top left pixel row: ")
                    else:
                        if (int(value)) >= 0 and (int(value)) < r:
                            is_int = True
                        else:
                            value = input("Error! Enter top left pixel row: ")
                        
                
                print("done 1")
                x0 = int(value)
                print("done 2")
                value = input("Enter top left pixel col: ")
                is_int = False
                while (is_int == False):
                    try:
                        # convert to integer
                        int(value)
                        
                    except ValueError:
                        is_int = False
                        value = input("Error! Enter top left pixel col: ")
                
                    else:
                        if (int(value)) >= 0 and (int(value)) < c:
                            is_int = True
                        else:
                            value = input("Error! Enter top left pixel col: ")
                        
                x1 = int(value)
                
                value = input("Enter bottom right pixel row: ")
                is_int = False
                while (is_int == False):
                    try:
                        # convert to integer
                        int(value)
                        
                    except ValueError:
                        is_int = False
                        value = input("Error! Enter bottom right pixel row: ")
                   
                    else:
                        if (int(value)) >= 0 and (int(value)) < r:
                            is_int = True
                        else:
                            value = input("Error! Enter top left pixel row: ")
                
                y0 = int(value)
                
                value = input("Enter bottom right pixel col: ")
                is_int = False
                while (is_int == False):
                    try:
                        # convert to integer
                        int(value)
                        
                    except ValueError:
                        is_int = False
                        value = input("Error! Enter bottom right pixel col: ")
                    
                    else:
                        if (int(value)) >= 0 and (int(value)) < c:
                            is_int = True
                        else:
                            value = input("Error! Enter bottom right pixel col: ")
                
                y1 = int(value)
                    
                x = (x0,x1)
                y = (y0,y1)
                
                if x0 >= y0 or x1 >= y1:
                    print("Error! Rectangle can't be formed")
                else: 
                    is_rectangle = True
                    
                    
            mask = rectangle_select(img, x, y)
        
        elif user_input == "8":
                
            r = len(img)
            c = len(img[0])
            
            value = input("Enter pixel row: ")
            is_int = False
            while (is_int == False):
                try:                    
                    int(value)
                        
                except ValueError:
                    is_int = False
                    value = input("Error! Enter pixel row: ")
                    
                else:
                    if (int(value)) >= 0 and (int(value)) < r:
                        is_int = True
                    else:
                        value = input("Error! Enter pixel row: ")           
                
            x0 = int(value)
            
            value = input("Enter pixel col: ")
            is_int = False
            while (is_int == False):
                try:                    
                    int(value)
                        
                except ValueError:
                    is_int = False
                    value = input("Error! Enter pixel col: ")
                    
                else:
                    if (int(value)) >= 0 and (int(value)) < c:
                        is_int = True
                    else:
                        value = input("Error! Enter pixel col: ")    
                                                
            x1 = int(value)     
            
            x = (x0,x1)
            
            thres = input("Enter threshold: ")
            is_int = False
            while (is_int == False):
                try: 
                    int(thres)
                        
                except ValueError:
                    is_int = False
                    thres = input("Error! threshold: ")
                    
                else:
                    if (int(thres)) >= 0:
                        is_int = True
                    else:
                        thres= input("Error! threshold: ")
            
            thres = int(thres)
            
            mask = magic_wand_select(img, x, thres)
        
        if np.all((mask == 1)) == 0:   
            for r in range(len(img)):
                for c in range(len(img[0])):
                    if mask[r][c] == 0:
                        for k in range(3):
                            img[r][c][k] = img2[r][c][k]
            
            
        display_image(img, mask)
                        
if __name__ == "__main__":
    menu()
