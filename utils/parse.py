import os

def threshold(arg):
    
    try:
        
        threshold = int(arg)
        
        return threshold
        
    except TypeError as e:
        
        print(e)

def frame_ignore(arg):
    
    try:
        
        frame_ignore = int(arg)
        
        return frame_ignore
        
    except TypeError as e:
        
        print(e)

def text_file(arg):
    
    if arg.endswith(".txt"):
        
        try:
            file = open(arg)
            file.close()
            return arg
        except FileNotFoundError as err:
            print(err)
            
    else:
        
        raise NameError("The file should be a .txt file")

def video_file(arg):
    
    if arg.endswith(".avi"):
        
        try:
            file = open(arg)
            file.close()
            return arg
        except FileNotFoundError as err:
            print(err)
            
    else:
        
        raise NameError("Needs to be an .avi file")
    
def h5_file(arg):
    
    if arg.endswith(".h5"):
        
        try:
            file = open(arg)
            file.close()
            return arg
        except FileNotFoundError as err:
            print(err)
            
    else:
        
        raise NameError("Needs to be an .h5 file")
    
def csv_file(arg):
    
    if arg.endswith(".csv"):
        
        try:
            file = open(arg)
            file.close()
            return arg
        except FileNotFoundError as err:
            print(err)
            
    else:
        
        raise NameError("Needs to be a .csv file")
    
def folder(arg):
    
    if os.path.isdir(arg):
        
        return arg
            
    else:
        
        raise NameError("Needs to be directory path")

    
