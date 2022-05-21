import sys
import numpy as np
from numpy.lib.stride_tricks import as_strided
sys.path.append('mytorch')
from loss import *
from activation import *
class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.state = x
        batch_size, in_channel, input_size = x.shape
        outputsize = (input_size - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.out_channel, outputsize))
        for batch_num in range(batch_size):
            for channel in range(self.out_channel):
                for i in range(outputsize):
                    # position needed to multiply
                    left_index = i * self.stride
                    right_index = left_index + self.kernel_size
                    output[batch_num,channel,i] = np.sum(np.multiply(x[batch_num,:,left_index:right_index], self.W[channel])) + self.b[channel]
        return output

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        dx = np.zeros(self.state.shape)
        self.db = np.sum(delta, (0, -1))
        batch_size, out_channel, output_size = delta.shape
        for i in range(output_size):
            left_index = i * self.stride
            right_index = left_index + self.kernel_size
            data_to_kernel = self.state[:,:,left_index:right_index]
            self.dW += np.dot(delta[:, :, i].T, data_to_kernel.reshape((batch_size, -1))).reshape(self.dW.shape) # delta: del_to_ker: bc, outsize, outchannel
            dx[:, :, left_index:right_index] += np.dot(delta[:, :, i], self.W.reshape((out_channel,-1))).reshape((data_to_kernel.shape))
        return dx


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride


        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.state = x
        batch_size, in_channel, input_width, input_height = x.shape
        kernel_size = self.kernel_size
        output_w = (input_width - kernel_size) // self.stride + 1
        output_h = (input_height - kernel_size) // self.stride + 1
        split = data_to_kernel_for2d(x, kernel_size, self.stride)
        output = np.tensordot(split, self.W,
                              axes=[(1, 4, 5), (1, 2, 3)]).transpose((0, 3, 1, 2))

        output += self.b.reshape(self.out_channel,1,1)
        return output

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        rotated_kernel = rotate_kernel(self.W)
        new_delta = dilationfor2d_refined(delta, self.stride,self.state.shape[-1], self.kernel_size)
        self.dW = np.tensordot(data_to_kernel_back2d(self.state, self.kernel_size), new_delta, axes=[(0, 2, 3), (0, 2, 3)]).transpose((3, 0, 1, 2))
        self.db = np.sum(delta, (0,2,3))
        new_delta = paddingfor2d(new_delta, self.kernel_size)
        dx = np.tensordot(data_to_kernel_back2d(new_delta,self.kernel_size), rotated_kernel, axes=[(1, 4, 5), (0, 2, 3)]).transpose((0, 3, 1, 2))
        # Batch_size, out_Channel, out_w, out_h, kernel_size, kernel_size
        # out_channel=2, in_channel=3, kernel_size=2, kernel_size=2
        return dx
        


class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Much like Conv2D, but take two attributes into consideration: padding and dilation.
        Make sure you have read the relative part in writeup and understand what we need to do here.
        HINT: the only difference are the padded input and dilated kernel.
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # After doing the dilation， the kernel size will be: (refer to writeup if you don't know)
        self.kernel_dilated = (self.kernel_size-1) * (self.dilation-1) + self.kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        # TODO: padding x with self.padding parameter (HINT: use np.pad())
        padded_x = paddingconvdil(x, self.padding)
        # TODO: do dilation -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        #       HINT: for loop to get self.W_dilated
        # self.W_dilated = dilationfor2d(self.W, self.dilation, self.kernel_size)
        self.W_dilated[:,:,::self.dilation,::self.dilation] = self.W
        # TODO: regular forward, just like Conv2d().forward()
        self.state = padded_x
        batch_size, in_channel, input_width, input_height = padded_x.shape
        kernel_size = self.kernel_dilated
        output_w = (input_width - kernel_size) // self.stride + 1
        output_h = (input_height - kernel_size) // self.stride + 1
        split = data_to_kernel_for2d(padded_x, kernel_size, self.stride) #
        output = np.tensordot(split, self.W_dilated,
                              axes=[(1, 4, 5), (1, 2, 3)]).transpose((0, 3, 1, 2))

        output += self.b.reshape(self.out_channel,1,1)
        return output



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        # TODO: main part is like Conv2d().backward(). The only difference are: we get padded input and dilated kernel
        #       for whole process while we only need original part of input and kernel for backpropagation.
        #       Please refer to writeup for more details.

        rotated_kernel = rotate_kernel(self.W_dilated)
        new_delta = dilationfor2d_refined(delta, self.stride, self.state.shape[-1], self.kernel_dilated)
        dw_dilated = np.tensordot(data_to_kernel_back2d(self.state, self.kernel_dilated), new_delta,
                               axes=[(0, 2, 3), (0, 2, 3)]).transpose((3, 0, 1, 2))
        # self.dW = np.tensordot(data_to_kernel_back2d(self.state, self.kernel_size), new_delta, axes=[(0, 2, 3), (0, 2, 3)]).transpose((3, 0, 1, 2))
        recover_dil = (self.kernel_size - 1) * (self.dilation - 1)
        # self.dW = dw_dilated[:,:,::recover_dil+1,::recover_dil+1]
        self.dW = dw_dilated[:, :, ::self.dilation, ::self.dilation]

        self.db = np.sum(delta, (0, 2, 3))
        new_delta = paddingfor2d(new_delta, self.kernel_dilated)#self.kernel_size
        # dx = np.tensordot(data_to_kernel_back2d(new_delta,self.kernel_size), rotated_kernel, axes=[(1, 4, 5), (0, 2, 3)]).transpose((0, 3, 1, 2))
        dx_dilated = np.tensordot(data_to_kernel_back2d(new_delta, self.kernel_dilated), rotated_kernel,
                          axes=[(1, 4, 5), (0, 2, 3)]).transpose((0, 3, 1, 2))
        recover_pad = self.padding
        dx = dx_dilated[:,:,recover_pad:-recover_pad,recover_pad:-recover_pad]
        # Batch_size, out_Channel, out_w, out_h, kernel_size, kernel_size
        # out_channel=2, in_channel=3, kernel_size=2, kernel_size=2
        return dx



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(self.b, -1)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape(self.b, self.c, self.w)



# 2d conv useful function
def rotate180(arr):
    rot_arr = arr.copy()
    rot_arr = rot_arr.reshape(arr.size)
    rot_arr = rot_arr[::-1]
    rot_arr = rot_arr.reshape(arr.shape)
    return rot_arr
def rotate_kernel(original_kernel):
    new_kernel = np.zeros(original_kernel.shape)
    for i in range(len(original_kernel)):
        for j in range(len(original_kernel[i])):
            new_kernel[i][j] = rotate180(original_kernel[i][j]).copy()
    return new_kernel
def dilationfor2d_refined(original_arr, stride,input_size,kernel_size):
    dilation = stride - 1
    if dilation == 0:
        return original_arr
    batch_size, channel, width, height = original_arr.shape
    dilated_data = original_arr.copy()
#     dilated_data = np.zeros((batch_size, channel, (width-1)*(dilation+1)+1, (height-1)*(dilation+1)+1))
    dilated_data = np.zeros((batch_size, channel, input_size-kernel_size+1, input_size-kernel_size+1))
    dilated_data[:, :, ::dilation + 1, ::dilation + 1] = original_arr
    return dilated_data
def paddingfor2d(original_arr, kernel_size):
    padding = kernel_size - 1
    padding_data = original_arr.copy()
    padding_data = np.pad(padding_data, ((0,0),(0,0),(padding,padding), (padding,padding)),'constant',constant_values=0)
    return padding_data

def data_to_kernel_for2d(x, kernel_size, stride):  # write these two into one if finish all before wed
    batch_size, input_channel, x_width, x_height = x.shape
    output_width = (x_width - kernel_size) // stride + 1
    output_height = (x_height - kernel_size) // stride + 1
    shape = (batch_size, input_channel, output_width, output_height, kernel_size, kernel_size)
    strides = (x.strides[0], x.strides[1], x.strides[-2] * stride,
               x.strides[-1] * stride, x.strides[-2], x.strides[-1])
    splitted_data = as_strided(x, shape, strides=strides)
    return splitted_data
def data_to_kernel_back2d(x, kernel_size, stride=1):  # write these two into one if finish all before wed
    batch_size, input_channel, x_width, x_height = x.shape
    output_width = (x_width - kernel_size) // stride + 1
    output_height = (x_height - kernel_size) // stride + 1
    shape = (batch_size, input_channel, output_width, output_height, kernel_size, kernel_size)
    strides = (x.strides[0], x.strides[1], x.strides[-2] * stride,
               x.strides[-1] * stride, x.strides[-2], x.strides[-1])
    splitted_data = as_strided(x, shape, strides=strides)
    return splitted_data

# dilation
def dilationfor2d(w, dilation, kernel_size):
    dil = (kernel_size - 1) * (dilation - 1)
    dilated_size = dil + kernel_size
    if dil == 0:
        return w
    out_channel, in_channel, kernel_wid, kernel_height = w.shape
    dilated_kernel = np.zeros((out_channel, in_channel, dilated_size, dilated_size))
    dilated_kernel[:, :, ::dil + 1, ::dil + 1] = w
    return dilated_kernel
def paddingconvdil(original_arr, padding):
    padding_data = original_arr.copy()
    padding_data = np.pad(padding_data, ((0,0),(0,0),(padding,padding), (padding,padding)),'constant',constant_values=0)
    return padding_data