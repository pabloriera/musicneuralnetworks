import numpy as np
import matplotlib.pyplot as plt
import os

def wavwrite(file_name,x,sr = 44100):
    from scipy.io.wavfile import write

    x = x/np.max(np.abs(x))*0.9
    write(file_name,sr,np.array(x*(2**15-1),dtype=np.int16))

def mp3write(file_name,x,sr = 44100):
    
    wavwrite(file_name,x,sr)
    wav2mp3(file_name)
    os.remove(file_name)


def wav2mp3(file_name,print_stdout=False):
    import subprocess

    command = ['ffmpeg','-y', '-i', file_name,'-b:a', '192k', file_name[:-4]+'.mp3']

    proc = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          stdin=subprocess.PIPE)
    if print_stdout:    
        print(proc.communicate())
    else:
        proc.communicate()


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    import numpy as np
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
    
def colorline( x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):

    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    import numpy as np
    import matplotlib.collections as mcoll

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc
    

def rc_default():
    import matplotlib as mpl
    mpl.rc('figure',figsize=(6,4))
    mpl.rc('figure',dpi=70)
    mpl.rc('legend',fontsize=8)
    mpl.rc('font',size=18)


def prepro(x, method='zscore', zca=True, epsilon = 0.01):
    from sklearn import preprocessing
    
    if method=='minmax':
        
        x = preprocessing.MinMaxScaler().fit_transform(x)
        
    elif method=='minabs':
        
        x = preprocessing.MinAbsScaler().fit_transform(x)
        
    elif method=='zscore':
        
        x = preprocessing.scale(x)
    
    if zca:
        return ZCA(regularization=epsilon).fit(x).transform(x)
    else:
        return x

class Descriptor():
    
    function = None
    data = None
    params = None
    data = None
    
    def __init__(self,data):
        self.name = data['name']
        self.params = data['params']
        self.function = data['function']
        
    def set_func(self,func):
        self.function=func
        
    def perform(self,input_data):
        self.data = self.function(self,input_data)
#         return self.descriptor

def audiofigure(func,duration, audio_path, sr = 44100, dpi=60, fps=4, ylim=(-1, 1),videoname=None,HTML_OUT=True):

    from matplotlib import animation, rc
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    from base64 import encodebytes
    import subprocess
    import os
    import tempfile
    
    t = np.arange(duration*sr)/float(sr)
    interval = 1000.0/fps
    frames = int(duration*fps)

    fig,ax = func()

    video_size = np.array(fig.get_size_inches())*dpi

    ax.set_xlim(( 0, duration))
    ax.set_ylim(ylim)

    line, = ax.plot([], [], color='r',lw=3)
    
    def init():
        line.set_data([], [])
        return (line,)

 
    def animate(i):
        x = np.array([i, i])/float(fps)
        y = ylim
        
        line.set_data(x, y)
        
        return (line,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=interval, blit=True);


    input_filename = tempfile.NamedTemporaryFile(delete=False).name+'.mp4'
    # print(input_filename)

    anim.save(input_filename,codec='h264')


    plt.close()

    if not videoname:
        videoname='out.mp4'

    command = ['ffmpeg', '-i', input_filename,'-y', '-i', audio_path ,'-c:v', 'libx264', '-c:a', 'libvorbis', 
               '-shortest', videoname]

    proc = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          stdin=subprocess.PIPE)
    proc.communicate()

    os.unlink(input_filename)
    
    if HTML_OUT:
        VIDEO_TAG = r'''<video {size} {options}>
          <source type="video/mp4" src="data:video/mp4;base64,{video}">
          Your browser does not support the video tag.
        </video>'''


        with open(videoname, 'rb') as video:
            vid64 = encodebytes(video.read())
            _base64_video = vid64.decode('ascii')
            _video_size = 'width="{0}" height="{1}"'.format(*video_size)

        options = ['controls', 'autoplay']
        
        #os.remove("in.mp4")
        #os.remove("out.mp4")

        html = VIDEO_TAG.format(video=_base64_video, size=_video_size, options=' '.join(options))
        return HTML(html)

from sklearn.base import TransformerMixin, BaseEstimator

class ZCA(BaseEstimator, TransformerMixin):
    
    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        from scipy import linalg    
        from sklearn.utils import as_float_array

        X = as_float_array(X, copy = self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        sigma = np.dot(X.T,X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed
    
def fragment_from_wav(filename,t1=0,t2=None):
    from scipy.io import wavfile
    import numpy as np
    fs,x = wavfile.read(filename)
    
    t1 = t1*fs
    if t2 is None:
        t2 = x.size
    else:
        t2 = t2*fs
        
    x = x[t1:t2]
    return fs,np.float64(x)/2**15
    
def lrelu(x, leak=0.2, name="lrelu"):
    import tensorflow as tf
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    
class Data():
    
    def __init__(self,data):
        self.data = data
        self.batch_ix = 0
        self.length = self.data.shape[0]
        self.ixs = np.arange(self.length)
        
    def next_batch(self,batch_size):
        np.random.shuffle(self.ixs)
        output = self.data[self.ixs[np.arange(self.batch_ix,self.batch_ix+batch_size) % self.length]]
        self.batch_ix+=batch_size
        return output


def audio2spectral(x,orig_fs=44100,resample_fs=22050,representation='STFT',magnitude=True,units='lineal',nfft_size= 2**10,nfft_hop=None,
             frame_size=64, step_size=None, n_bins = 84,normalization_axis=None):
    
    from scipy.signal import resample
    import librosa
    import numpy as np

    # Read wav file to floating values
   
    fs = resample_fs
    x = resample(x, int(x.size*fs/orig_fs))
       
    # Peak Normalization
    x/=abs(x).max()

    if step_size is None:
        step_size = int(frame_size/2)
        if step_size==0:
            step_size=1
            
    if nfft_hop is None:
        nfft_hop = int(nfft_size/2)
        
    
    if representation=='STFT':
        # STFT
        S = librosa.stft(x,n_fft=nfft_size,hop_length=nfft_hop,win_length=nfft_size )/2/nfft_size

    elif representation=='CQT':
        # CQT
        S = librosa.cqt(x,sr=fs,hop_length=nfft_hop,fmin=40.0,n_bins=n_bins,real=False)

    S = S[::-1,:]
    
    if magnitude:
        S = abs(S)
        
    if units=='db':
        S = abs(S)        
        S = 20*np.log10(S/S.max()).clip(-60,0)

    S = (S - S.min(normalization_axis)) /(S.max(normalization_axis) - S.min(normalization_axis))
    
    n_frames = int( (S.shape[1]-frame_size)/step_size+1 )
       
    return np.array( [S[:,i*step_size:i*step_size+frame_size] for i in range(n_frames)] )

def montage(images):
    """Draw all images as a montage separated by 1 pixel borders.

    Parameters
    ----------
    images : numpy.ndarray
        Input array to create montage of.  Array should be:
        batch x height x width x channels.
    
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    else:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m

def framesoverlap(frames,hop,win_lenght,window=None):
    from scipy.signal import hanning

    if not window:
        window = hanning(win_lenght,sym=False)

    ntotal = (frames.shape[1]-1)*hop + win_lenght
        
    outsignal = np.zeros(ntotal)

    k=-1
    npos = 0
    while npos<ntotal:
        if (ntotal-npos)>win_lenght:
            k+=1
            outsignal[npos:npos+win_lenght] += frames[:,k]*window
            npos += hop

        else:
            break
            
    return outsignal


def dict_product_args(D,func):
    import inspect
    import itertools
    args = inspect.getargspec(func).args
    defaults = inspect.getargspec(func).defaults

    for i,k in enumerate(args[-len(defaults):]):
        if k not in D.keys():
            D[k]=[defaults[i]]

    combis = []
    for combination in itertools.product(*D.values()):
        combis.append(dict(zip(D.keys(),combination)))

    return combis

def axes3d():
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.gcf()
    if plt.gca().__class__.__name__=='AxesSubplot':
        ax = fig.add_subplot(*plt.gca().properties()['geometry'], projection='3d')
    elif plt.gca().__class__.__name__=='Axes3DSubplot':
        ax = plt.gca()
    else:
        ax = fig.add_subplot(111, projection='3d')

    return ax

def plot3(x,y,z,*args,**kwargs):

    ax = axes3d()
    
    ax.plot3D(x,y,z,*args,**kwargs)
    plt.draw()
    
    return ax