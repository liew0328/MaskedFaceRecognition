import os
import uuid
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.properties import StringProperty
from kivy.uix.screenmanager import Screen
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from torch.utils.data import DataLoader
from torchvision import datasets


class VerificationScreen(Screen):
    status = StringProperty()

    def __init__(self, **kwargs):
        super(VerificationScreen, self).__init__(**kwargs)
        self.status = 'Ready for verification'

    def verify(self):
        cam = FaceRecognizeApp
        ret, frame = cam.cap.read()
        # convert image from BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
        # detect face with MTCNN
        face, prob = mtcnn(img, return_prob=True)
        if face is None:
            self.status = "No face detected"
            return
        # convert face into embedding
        net = InceptionResnetV1(pretrained='vggface2').eval()
        emb = net(face.unsqueeze(0)).detach()  # detach is to make required gradient false
        # load the embedding file
        faceEmbedding = torch.load('embedding_data.pt')
        embeddingList = faceEmbedding[0]
        nameList = faceEmbedding[1]
        # list of matched distances, minimum distance is used to identify the person
        distList = []
        for idx, embeddingDF in enumerate(embeddingList):
            dist = torch.dist(emb,
                              embeddingDF).item()  # dist returns p-norm, item return standard Python number from tensor
            distList.append(dist)
        idx_min = distList.index(min(distList))
        # if the minimum distance is lesser than the threshold then only would be recognized as the person
        if min(distList) < 0.8:
            self.status = 'Matched: ' + nameList[idx_min]
        else:  # the person is not in the DF
            self.status = 'Unknown'
        Logger.info('Matched with: ' + nameList[idx_min] + ' With distance: ' + str("{0:.4f}".format(min(distList))))


imagePath = ''


class RegistrationScreen(Screen):
    dialog = None

    def register(self):
        global imagePath
        # read from the txtUsername
        userName = self.ids.txtUsername.text
        # create a folder with userName
        imagePath = os.path.join('FaceImages', str(userName))
        try:
            os.makedirs(imagePath)
        except FileExistsError:  # if the userName folder existed
            # show a warning dialog
            if not self.dialog:
                self.dialog = MDDialog(
                    title='User Existed!',
                    text='The user name existed, do you want to capture new photo?',
                    buttons=[
                        MDFlatButton(text='No',
                                     on_release=self.closeDialog),
                        MDFlatButton(text='Yes',
                                     on_release=self.changeScreen)
                    ]
                )
            self.dialog.open()
            return
        except OSError as error:  # exit the application if other errors occurred
            Logger.error(error)
            exit(1)
        self.manager.current = 'screenCaptureFace'

    # change the current screen
    def changeScreen(self, obj):
        self.dialog.dismiss()
        self.manager.current = 'screenCaptureFace'

    # close the warning dialog
    def closeDialog(self, obj):
        self.dialog.dismiss()


class CaptureFaceScreen(Screen):
    # save the image to the folder created in registrationScreen
    def capture(self):
        global imagePath
        cam = self.ids['camera']
        # Naming out image path
        imgname = os.path.join(imagePath, str(uuid.uuid1()) + '.jpg')
        cam.export_to_png(imgname)
        Logger.info("Captured: " + imgname)


class TrainingScreen(Screen):
    def train(self):
        Logger.info("Initializing...")
        # initialize MTCNN & pretrained InceptionResnet
        mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
        net = InceptionResnetV1(pretrained='vggface2').eval()
        # face image folder path
        dataset = datasets.ImageFolder('FaceImages')
        # get users' name from folder names
        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}  # Dictionary mapping class name to class index.

        def collate_fn(x):
            return x[0]

        loader = DataLoader(dataset, collate_fn=collate_fn)
        # collate_fn -> merges a list of samples to form a mini-batch of Tensor(s).
        # Used when using batched loading from a map-style dataset.
        name_list = []
        embedding_list = []
        Logger.info("Training...")
        for img, idx in loader:
            face, prob = mtcnn(img, return_prob=True)
            # if face detected and probability more than 90%
            if face is not None and prob > 0.90:
                # convert face image into embedding
                emb = net(face.unsqueeze(0))
                embedding_list.append(emb.detach())
                name_list.append(idx_to_class[idx])
        data = [embedding_list, name_list]
        # save embedding
        torch.save(data, 'embedding_data.pt')
        Logger.info("Training done...")


class FaceRecognizeApp(MDApp):
    # OpenCV - access webcam
    cap = cv2.VideoCapture(1)

    def build(self):
        return Builder.load_file("FaceRecognize.kv")


if __name__ == '__main__':
    FaceRecognizeApp().run()
