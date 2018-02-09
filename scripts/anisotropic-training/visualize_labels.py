
import skimage.measure
import matplotlib.patheffects as path_effects

def plot_label_colors(width=0.3, features=FEATURES, colors=COLORS):
    ax = gca()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    width *= xmax-xmin
    
    ax.set_xlim(xmin, xmax+width)
   
    x = xmax
    h = (ymax-ymin)/len(features)
    for i, label in enumerate(FEATURES):
        y = ymin+i*h
        R = Rectangle((x,y), width, h, fill=True, color=COLORS[i], alpha=1)
        ax.add_patch(R)
        R = Rectangle((x,y), width, h, fill=False, color=0.5*COLORS[i], alpha=1 )
        ax.add_patch(R)
        text(x+0.5*width, y+0.5*h, label, color='black', 
             horizontalalignment="center", verticalalignment="center")

    
def plot_mask(mask, color, edge_color=None, unknown_color=None, alpha=0.7):
    color = array(color)
    color[3] *= alpha
    
    if edge_color is None:
        edge_color = array(color)
        edge_color[3] = 1
        
    if unknown_color is None:
        unknown_color = array([0.5]*4)
        
        
    colors = array([np.zeros_like(color), unknown_color, color, edge_color])[mask.astype(int)]
    imshow(colors)
    

def plot_labels(mask, color, label, fontsize=8):
    regions = skimage.measure.regionprops(skimage.measure.label(mask>1))
    for r in regions:
        y0, x0 = r.centroid
        text(x0, y0, label, color=color, fontdict={'weight':'bold'},
             size=fontsize, ha='center', va='center',
             path_effects=[
                path_effects.withSimplePatchShadow(shadow_rgbFace=(0,0,0), alpha=1),
                path_effects.Stroke(linewidth=2, foreground='white'),
                path_effects.Normal()
            ])
        
   
def plot_all_expected(expected, features=FEATURES, colors=COLORS, edge_colors=None, labels=True):
    if hasattr(expected, 'keys'):
        if 'unlabeled' in expected:
            plot_mask(expected['unlabeled'], color=array([0.5]*4), unknown_color=array([0.5]*4))
        for i in range(len(features)):
            if features[i] in expected:
                mask = array(expected[features[i]])
                plot_mask(expected[features[i]], color=colors[i])
        if labels:
            if 'unlabeled' in expected:
                mask = array(expected['unlabeled']+1)
                plot_labels(mask, color=array([0.5]*3), label='unlabeled')            

            for i in range(len(features)):
                if features[i] in expected:
                    mask = array(expected[features[i]])
                    plot_labels(mask, color=colors[i, :3], label=features[i])            
    else:
        for i in range(len(features)):
            plot_mask(expected[i], color=colors[i])
        if labels:
            for i in range(len(features)):
                mask = array(expected[i])
                plot_labels(mask, color=colors[i,:3], label=features[i])


def visualize_expected(array_or_path, legend=True, legend_width=0.3, labels=True):
    if isinstance(array_or_path, str):
        data = np.load(array_or_path)
    else:
        data = array_or_path
    if hasattr(data, 'keys'):
        rgb = data['rgb']/255.
        _labels = data
    else:
        rgb = data[:3].transpose(1,2,0)/255.
        _labels = data[3:]
    imshow(rgb)
    plot_all_expected(_labels, labels=labels)
    xticks([]); yticks([])
    if legend:
        plot_label_colors(width=legend_width)


#visualize_expected(example_path)
#visualize_expected('./data/aeriels-24class/regent_many-0421-facade-02-original/01/0001.npz')
#visualize_expected('./data/aeriels-24class/regent_many-0418-facade-02-original/01/0001.npz')