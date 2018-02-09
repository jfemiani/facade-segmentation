
ERROR_COLORS = [[128, 128,  128], #TN
                [255, 0,    0],   #FP
                [0,   255,  0],   #TP
                [0,   0,    255], #FN
                [0,   0,    0], #ignored
               ]  
ERROR_COLORS = array(ERROR_COLORS, dtype=np.uint8)

def color_coded_errors(expected, predicted, ignored=None, colors=ERROR_COLORS):
    TP = ~ignored & (expected & predicted)
    FN = ~ignored & (expected & ~predicted)
    FP = ~ignored & (~expected & predicted)
    TN = ~ignored & (~expected & ~predicted)
    errors = np.argmax(array([TN, FP, TP, FN, ignored], dtype=np.uint8), axis=0)
    if colors is not None:
        return np.ma.masked_array(colors[errors], np.dstack([TN]*3))
    else:
        return errors  
      
def render_errors(path, alpha=0.5, noFN=True):
    mf, rgb, expected, predicted = get_metrics(path)
    ignored = (expected != mf.label_positive) & (expected != mf.label_negative)
    expected = expected==mf.label_positive
    predicted = predicted==mf.label_positive
    print mf.label_positive
    cc = color_coded_errors(expected, predicted, ignored)
    rgb = rgb.transpose(1,2,0)/255.
    if noFN:
        rgb[~cc.mask] = (1-alpha)*rgb[~cc.mask] + alpha*cc[~cc.mask]
        return rgb.clip(0,1)
        #imshow(predicted)
    else:
        rgb = (1-alpha)*rgb + alpha*cc
    return rgb