
import numpy as np
import time
import paddle.fluid as fluid
from PIL import Image
from PIL import ImageDraw

def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = Image.open(img_path)
    return origin




class inference():
    def __init__(self):


        train_parameters = {
        "label_dict": {0: "apple", 1: "banana", 2: "orange"},
        "use_gpu": True,
        "input_size": [3, 608, 608],  # 原版的边长大小为608，为了提高训练速度和预测速度，此处压缩为448
    }

        self.target_size = train_parameters['input_size']

        self.label_dict = train_parameters['label_dict']

    # class_dim = train_parameters['class_dim']
    # print("label_dict:{} class dim:{}".format(label_dict, class_dim))
        place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
        self.exe = fluid.Executor(place)
        path = "C:\\Users\\zhili\\Desktop\\infer-paddle\\model"  # _mobilenet_v1
        [self.inference_program, self.feed_target_names, self.fetch_targets] = fluid.io.load_inference_model \
        (dirname=path, executor=self.exe, model_filename='__model__', params_filename='__params__')


    def draw_bbox_image(self,img, boxes, labels,scores):
        """
        给图片画上外接矩形框
        :param img:
        :param boxes:
        :param save_name:
        :param labels
        :return:
        """
        draw = ImageDraw.Draw(img)
        for box, label,score in zip(boxes, labels,scores):
            print(box, label, score)
            if(score >0.3):
                xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                draw.rectangle((xmin, ymin, xmax, ymax), None, 'red')
                draw.text((xmin, ymin), self.label_dict[label], (255, 255, 0))
        # img.save(save_name)

    def resize_img(self,img, target_size):#将图片resize到target_size
        """
        保持比例的缩放图片
        :param img:
        :param target_size:
        :return:
        """
        img = img.resize(target_size[1:], Image.BILINEAR)
        return img

    def tensor(self, img):
        img = self.resize_img(img, self.target_size)
        resized_img = img.copy()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW

        img = img / 255.0

        img[0, :, :] -= 0.485
        img[1, :, :] -= 0.456
        img[2, :, :] -= 0.406

        img[0, :, :] /= 0.229
        img[1, :, :] /= 0.224
        img[2, :, :] /= 0.225
        tensor_img = img[np.newaxis, :]
        return tensor_img



    def infer(self,origin):
        """
        预测，将结果保存到一副新的图片中
        :param image_path:
        :return:
        """

        input_w, input_h = origin.size[0], origin.size[1]
        image_shape = np.array([input_h, input_w], dtype='int32')
        t1 = time.time()
        tensor_img = self.tensor(origin)
        batch_outputs = self.exe.run(self.inference_program,
                                feed={self.feed_target_names[0]: tensor_img,
                                      self.feed_target_names[1]: image_shape[np.newaxis, :]},
                                fetch_list=self.fetch_targets,
                                return_numpy=False)

        period = time.time() - t1
        print("predict cost time:{0}".format("%2.2f sec" % period))
        bboxes = np.array(batch_outputs[0])

        if bboxes.shape[1] != 6:
            print("No object found in {}".format(image_path))
            return
        labels = bboxes[:, 0].astype('int32')
        scores = bboxes[:, 1].astype('float32')
        boxes = bboxes[:, 2:].astype('float32')

        # last_dot_index = image_path.rfind('.')
        # out_path = image_path[:last_dot_index]
        # out_path += '-result.jpg'
        self.draw_bbox_image(origin, boxes, labels,scores)
        return origin



if __name__ == '__main__':
    img_path= "C:\\Users\\zhili\\Desktop\\infer-paddle\\pic\\2.jpg"
    origin = read_image(img_path)
    a=inference()
    a.infer(origin)