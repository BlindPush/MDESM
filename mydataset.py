import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import random
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

class Dataset5(Dataset):
    def __init__(self, img_dir, idlist, phase):
        self.img_dir = img_dir
        self.idlist = idlist
        self.phase = phase
    def __getitem__(self, item):
        self.id = self.idlist[item]
        self.datapatht2i = os.path.join(self.img_dir, 'T2', self.id)
        self.datapathdwii = os.path.join(self.img_dir, 'DWI', self.id)
        self.datapathadci = os.path.join(self.img_dir, 'ADC', self.id)

        t2h5 = h5py.File(self.datapatht2i, 'r')
        dwih5 = h5py.File(self.datapathdwii, 'r')
        adch5 = h5py.File(self.datapathadci, 'r')

        t2img = np.asarray(t2h5['image'])
        dwiimg = np.asarray(dwih5['image'])
        adcimg = np.asarray(adch5['image'])

        self.img = np.concatenate([t2img, dwiimg, adcimg], axis=2)
        if self.phase == 'train':
            self.imgori, self.img = self.transforms_train(self.img)
        else:
            self.imgori, self.img = self.transforms_test(self.img)

        self.imgori = torch.tensor(self.imgori).float()
        self.img = torch.tensor(self.img).float()
        self.label = torch.tensor(np.array(t2h5['label'])).long()
        self.labelpsa = torch.tensor(np.array(t2h5['psa'])).float()
        self.labelpirads = torch.tensor(np.array(t2h5['pirads'])).float()
        self.labeldmax = torch.tensor(np.array(t2h5['dmax'])).float()
        self.labeladcmean = torch.tensor(np.array(t2h5['adcmean'])).float()
        self.labelage = torch.tensor(np.array(t2h5['age'])).float()

        self.cli = [self.labelpsa, self.labelage, self.labelpirads,
                    self.labeldmax, self.labeladcmean]
        self.cli = torch.tensor(self.cli)
        return self.id, self.imgori, self.img, self.label, self.cli
    def __len__(self):
        return len(self.idlist)

    def standardize(self, image):
        im = image
        im = (im - im.min()) / (im.max() - im.min())
        return im

    def transforms_train(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.99, saturation=0.9, hue=0., p=0.5),
            A.GaussNoise(p=0.2),
        ])
        img = train_trans(image=image)['image']

        image_mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
        image_std = np.array([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])])
        imgn = (img-image_mean)/(image_std+1e-10)
        imgori = imgn * (image_std + 1e-10) + image_mean

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img
    def transforms_test(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
        ])

        img = image
        image_mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
        image_std = np.array([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])])
        imgn = (img - image_mean) / (image_std + 1e-10)
        imgori = imgn * (image_std + 1e-10) + image_mean

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img

class Dataset4(Dataset):
    def __init__(self, img_dir, idlist, phase):
        self.img_dir = img_dir
        self.idlist = idlist
        self.phase = phase
    def __getitem__(self, item):
        self.id = self.idlist[item]
        self.datapatht2i = os.path.join(self.img_dir, 'T2', self.id)
        self.datapathdwii = os.path.join(self.img_dir, 'DWI', self.id)
        self.datapathadci = os.path.join(self.img_dir, 'ADC', self.id)

        t2h5 = h5py.File(self.datapatht2i, 'r')
        dwih5 = h5py.File(self.datapathdwii, 'r')
        adch5 = h5py.File(self.datapathadci, 'r')

        t2img = np.asarray(t2h5['image'])
        dwiimg = np.asarray(dwih5['image'])
        adcimg = np.asarray(adch5['image'])

        self.img = np.concatenate([t2img, dwiimg, adcimg], axis=2)
        if self.phase == 'train':
            self.imgori, self.img = self.transforms_train(self.img)
            self.fre = np.fft.fft2(self.img, axes=(1, 2))
            self.fre_p = np.angle(self.fre)
            self.fre_m = np.e ** (1j * self.fre_p)
            self.img_onlyphase = np.abs(np.fft.ifft2(self.fre_m, axes=(1, 2)))
        else:
            self.imgori, self.img = self.transforms_test(self.img)
            self.fre = np.fft.fft2(self.img, axes=(1, 2))
            self.fre_p = np.angle(self.fre)
            self.fre_m = np.e ** (1j * self.fre_p)
            self.img_onlyphase = np.abs(np.fft.ifft2(self.fre_m, axes=(1, 2)))

        self.imgori = torch.tensor(self.imgori).float()
        self.img = torch.tensor(self.img).float()
        self.label = torch.tensor(np.array(t2h5['label'])).long()
        self.labelpsa = torch.tensor(np.array(t2h5['psa'])).float()
        self.labeldmax = torch.tensor(np.array(t2h5['dmax'])).float()
        self.labeladcmean = torch.tensor(np.array(t2h5['adcmean'])).float()
        self.labelage = torch.tensor(np.array(t2h5['age'])).float()

        self.cli = [self.labelpsa, self.labelage, self.labeldmax, self.labeladcmean]
        self.cli = torch.tensor(self.cli)
        return self.id, self.imgori, self.img, self.label, self.cli
    def __len__(self):
        return len(self.idlist)

    def standardize(self, image):
        im = image
        im = (im - im.min()) / (im.max() - im.min())
        return im

    def transforms_train(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.99, saturation=0.9, hue=0., p=0.5),
            A.GaussNoise(var_limit=(0.0, 0.001), p=0.2),
        ])
        img = train_trans(image=image)['image']

        image_mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
        image_std = np.array([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])])
        imgn = (img-image_mean)/(image_std+1e-10)
        imgori = imgn * (image_std + 1e-10) + image_mean

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img
    def transforms_test(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
        ])

        img = image
        image_mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
        image_std = np.array([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])])
        imgn = (img - image_mean) / (image_std + 1e-10)
        imgori = imgn * (image_std + 1e-10) + image_mean


        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img


class Dataset4v2(Dataset):
    def __init__(self, img_dir, idlist, phase):
        self.img_dir = img_dir
        self.idlist = idlist
        self.phase = phase
    def __getitem__(self, item):
        self.id = self.idlist[item]
        self.datapatht2i = os.path.join(self.img_dir, 'T2', self.id)
        self.datapathdwii = os.path.join(self.img_dir, 'DWI', self.id)
        self.datapathadci = os.path.join(self.img_dir, 'ADC', self.id)

        t2h5 = h5py.File(self.datapatht2i, 'r')
        dwih5 = h5py.File(self.datapathdwii, 'r')
        adch5 = h5py.File(self.datapathadci, 'r')

        t2img = np.asarray(t2h5['image'])
        dwiimg = np.asarray(dwih5['image'])
        adcimg = np.asarray(adch5['image'])

        self.img = np.concatenate([t2img, dwiimg, adcimg], axis=2)
        if self.phase == 'train':
            self.imgori, self.img = self.transforms_train(self.img)
        else:
            self.imgori, self.img = self.transforms_test(self.img)

        self.imgori = torch.tensor(self.imgori).float()
        self.img = torch.tensor(self.img).float()
        self.label = torch.tensor(np.array(t2h5['label'])).long()
        self.labelpsa = torch.tensor(np.array(t2h5['psa'])).float()
        self.labeldmax = torch.tensor(np.array(t2h5['dmax'])).float()
        self.labeladcmean = torch.tensor(np.array(t2h5['adcmean'])).float()
        self.labelage = torch.tensor(np.array(t2h5['age'])).float()

        self.cli = [self.labelpsa, self.labelage, self.labeldmax, self.labeladcmean]
        self.cli = torch.tensor(self.cli)
        return self.id, self.imgori, self.img, self.label, self.cli
    def __len__(self):
        return len(self.idlist)

    def standardize(self, image):
        im = image
        im = (im - im.min()) / (im.max() - im.min())
        return im

    def transforms_train(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.99, saturation=0.9, hue=0., p=0.5),
            A.GaussNoise(var_limit=(0.0, 0.001), p=0.2),
        ])
        img = train_trans(image=image)['image']

        imgn = img
        imgori = img

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img
    def transforms_test(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
        ])

        img = image
        imgn = img
        imgori = img


        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img

class Datasetdom(Dataset):
    def __init__(self, img_dir, idlist, domainlabel, phase):
        self.img_dir = img_dir
        self.idlist = idlist
        self.domainlabel = domainlabel
        self.phase = phase
    def __getitem__(self, item):
        self.id = self.idlist[item]
        self.datapatht2i = os.path.join(self.img_dir, 'T2', self.id)
        self.datapathdwii = os.path.join(self.img_dir, 'DWI', self.id)
        self.datapathadci = os.path.join(self.img_dir, 'ADC', self.id)

        t2h5 = h5py.File(self.datapatht2i, 'r')
        dwih5 = h5py.File(self.datapathdwii, 'r')
        adch5 = h5py.File(self.datapathadci, 'r')

        t2img = np.asarray(t2h5['image'])
        dwiimg = np.asarray(dwih5['image'])
        adcimg = np.asarray(adch5['image'])

        self.img = np.concatenate([t2img, dwiimg, adcimg], axis=2)
        if self.phase == 'train':
            self.imgori, self.img = self.transforms_train(self.img)
        else:
            self.imgori, self.img = self.transforms_test(self.img)

        self.imgori = torch.tensor(self.imgori).float()
        self.img = torch.tensor(self.img).float()
        self.label = torch.tensor(np.array(t2h5['label'])).long()
        self.labelpsa = torch.tensor(np.array(t2h5['psa'])).float()
        self.labelpirads = torch.tensor(np.array(t2h5['pirads'])).float()
        self.labeldmax = torch.tensor(np.array(t2h5['dmax'])).float()
        self.labeladcmean = torch.tensor(np.array(t2h5['adcmean'])).float()
        self.labelage = torch.tensor(np.array(t2h5['age'])).float()
        self.dom = torch.tensor(self.domainlabel).long()

        self.cli = [self.labelpsa, self.labelage, self.labelpirads,
                    self.labeldmax, self.labeladcmean]
        self.cli = torch.tensor(self.cli)
        return self.id, self.imgori, self.img, self.label, self.cli, self.dom
    def __len__(self):
        return len(self.idlist)

    def transforms_train(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.99, saturation=0.9, hue=0., p=0.5),
            A.GaussNoise(p=0.2),
        ])
        img = train_trans(image=image)['image']

        image_mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
        image_std = np.array([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])])
        imgn = (img-image_mean)/(image_std+1e-10)
        imgori = imgn * (image_std + 1e-10) + image_mean

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img
    def transforms_test(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
        ])

        img = image
        image_mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
        image_std = np.array([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])])
        imgn = (img - image_mean) / (image_std + 1e-10)
        imgori = imgn * (image_std + 1e-10) + image_mean

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img

class Datasetdom4(Dataset):
    def __init__(self, img_dir, idlist, domainlabel, phase):
        self.img_dir = img_dir
        self.idlist = idlist
        self.domainlabel = domainlabel
        self.phase = phase
    def __getitem__(self, item):
        self.id = self.idlist[item]
        self.datapatht2i = os.path.join(self.img_dir, 'T2', self.id)
        self.datapathdwii = os.path.join(self.img_dir, 'DWI', self.id)
        self.datapathadci = os.path.join(self.img_dir, 'ADC', self.id)

        t2h5 = h5py.File(self.datapatht2i, 'r')
        dwih5 = h5py.File(self.datapathdwii, 'r')
        adch5 = h5py.File(self.datapathadci, 'r')

        t2img = np.asarray(t2h5['image'])
        dwiimg = np.asarray(dwih5['image'])
        adcimg = np.asarray(adch5['image'])

        self.img = np.concatenate([t2img, dwiimg, adcimg], axis=2)
        if self.phase == 'train':
            self.imgori, self.img = self.transforms_train(self.img)
        else:
            self.imgori, self.img = self.transforms_test(self.img)

        self.imgori = torch.tensor(self.imgori).float()
        self.img = torch.tensor(self.img).float()
        self.label = torch.tensor(np.array(t2h5['label'])).long()
        self.labelpsa = torch.tensor(np.array(t2h5['psa'])).float()
        self.labeldmax = torch.tensor(np.array(t2h5['dmax'])).float()
        self.labeladcmean = torch.tensor(np.array(t2h5['adcmean'])).float()
        self.labelage = torch.tensor(np.array(t2h5['age'])).float()
        self.dom = torch.tensor(self.domainlabel).long()

        self.cli = [self.labelpsa, self.labelage, self.labeldmax, self.labeladcmean]
        self.cli = torch.tensor(self.cli)
        return self.imgori, self.img, self.label, self.cli, self.dom
    def __len__(self):
        return len(self.idlist)

    def transforms_train(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.99, saturation=0.9, hue=0., p=0.5)
        ])
        img = train_trans(image=image)['image']

        image_mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
        image_std = np.array([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])])
        imgn = (img-image_mean)/(image_std+1e-10)
        imgori = imgn * (image_std + 1e-10) + image_mean

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img
    def transforms_test(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
        ])

        img = image
        image_mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
        image_std = np.array([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])])
        imgn = (img - image_mean) / (image_std + 1e-10)
        imgori = imgn * (image_std + 1e-10) + image_mean

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img

class Datasetdom4v2(Dataset):
    def __init__(self, img_dir, idlist, img_dir2, idlist2, phase):
        self.img_dir = img_dir
        self.idlist = idlist
        self.img_dir2 = img_dir2
        self.idlist2 = idlist2
        self.phase = phase

        zhenghelist = self.zhenghe(self.img_dir, self.idlist, self.img_dir2, self.idlist2)
        self.zht2i, self.zhdwii, self.zhadci = zhenghelist

    def __getitem__(self, item):
        self.datapatht2i = self.zht2i[item][0]
        self.datapathdwii = self.zhdwii[item][0]
        self.datapathadci = self.zhadci[item][0]
        self.domainlabel = self.zht2i[item][1]

        t2h5 = h5py.File(self.datapatht2i, 'r')
        dwih5 = h5py.File(self.datapathdwii, 'r')
        adch5 = h5py.File(self.datapathadci, 'r')

        t2img = np.asarray(t2h5['image'])
        dwiimg = np.asarray(dwih5['image'])
        adcimg = np.asarray(adch5['image'])

        self.img = np.concatenate([t2img, dwiimg, adcimg], axis=2)
        if self.phase == 'train':
            self.imgori, self.img = self.transforms_train(self.img)
        else:
            self.imgori, self.img = self.transforms_test(self.img)

        self.imgori = torch.tensor(self.imgori).float()
        self.img = torch.tensor(self.img).float()
        self.label = torch.tensor(np.array(t2h5['label'])).long()
        self.labelpsa = torch.tensor(np.array(t2h5['psa'])).float()
        self.labeldmax = torch.tensor(np.array(t2h5['dmax'])).float()
        self.labeladcmean = torch.tensor(np.array(t2h5['adcmean'])).float()
        self.labelage = torch.tensor(np.array(t2h5['age'])).float()
        self.dom = np.array(self.domainlabel)

        self.cli = [self.labelpsa, self.labelage, self.labeldmax, self.labeladcmean]
        self.cli = torch.tensor(self.cli)
        return self.imgori, self.img, self.label, self.cli, self.dom
    def __len__(self):
        return len(self.idlist+self.idlist2)

    def transforms_train(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.99, saturation=0.9, hue=0., p=0.5)
        ])
        img = train_trans(image=image)['image']

        image_mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
        image_std = np.array([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])])
        imgn = (img-image_mean)/(image_std+1e-10)
        imgori = imgn * (image_std + 1e-10) + image_mean

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img
    def transforms_test(self, image):
        imgmin = np.array([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
        imgmax = np.array([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
        ])

        img = image
        image_mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])
        image_std = np.array([np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])])
        imgn = (img - image_mean) / (image_std + 1e-10)
        imgori = imgn * (image_std + 1e-10) + image_mean

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img

    def zhenghe(self, img_dir, idlist, img_dir2, idlist2):
        dom1t2pth = []
        dom1dwipth = []
        dom1adcpth = []
        dom2t2pth = []
        dom2dwipth = []
        dom2adcpth = []
        for d1 in idlist:
            dom1t2pth.append([os.path.join(img_dir, 'T2', d1), 0])
            dom1dwipth.append([os.path.join(img_dir, 'DWI', d1), 0])
            dom1adcpth.append([os.path.join(img_dir, 'ADC', d1), 0])
        for d2 in idlist2:
            dom2t2pth.append([os.path.join(img_dir2, 'T2', d2), 1])
            dom2dwipth.append([os.path.join(img_dir2, 'DWI', d2), 1])
            dom2adcpth.append([os.path.join(img_dir2, 'ADC', d2), 1])
        domt2pth = dom1t2pth + dom2t2pth
        domdwipth = dom1dwipth + dom2dwipth
        domadcpth = dom1adcpth + dom2adcpth

        return domt2pth, domdwipth, domadcpth

class Datasetdom4v3(Dataset):
    def __init__(self, img_dir, idlist, img_dir2, idlist2, phase):
        self.img_dir = img_dir
        self.idlist = idlist
        self.img_dir2 = img_dir2
        self.idlist2 = idlist2
        self.phase = phase
        zhenghelist = self.zhenghe(self.img_dir, self.idlist, self.img_dir2, self.idlist2)
        self.zht2i, self.zhdwii, self.zhadci = zhenghelist

    def __getitem__(self, item):
        self.datapatht2i = self.zht2i[item][0]
        self.datapathdwii = self.zhdwii[item][0]
        self.datapathadci = self.zhadci[item][0]
        self.domainlabel = self.zht2i[item][1]

        t2h5 = h5py.File(self.datapatht2i, 'r')
        dwih5 = h5py.File(self.datapathdwii, 'r')
        adch5 = h5py.File(self.datapathadci, 'r')

        t2img = np.asarray(t2h5['image'])
        dwiimg = np.asarray(dwih5['image'])
        adcimg = np.asarray(adch5['image'])

        self.img = np.concatenate([t2img, dwiimg, adcimg], axis=2)
        if self.phase == 'train':
            self.imgori, self.img = self.transforms_train(self.img)
        else:
            self.imgori, self.img = self.transforms_test(self.img)

        self.imgori = torch.tensor(self.imgori).float()
        self.img = torch.tensor(self.img).float()
        self.label = torch.tensor(np.array(t2h5['label'])).long()
        self.labelpsa = torch.tensor(np.array(t2h5['psa'])).float()
        self.labeldmax = torch.tensor(np.array(t2h5['dmax'])).float()
        self.labeladcmean = torch.tensor(np.array(t2h5['adcmean'])).float()
        self.labelage = torch.tensor(np.array(t2h5['age'])).float()
        self.dom = torch.tensor(self.domainlabel).long()

        self.cli = [self.labelpsa, self.labelage, self.labeldmax, self.labeladcmean]
        self.cli = torch.tensor(self.cli)
        return self.imgori, self.img, self.label, self.cli, self.dom
    def __len__(self):
        return len(self.idlist+self.idlist2)

    def transforms_train(self, image):
        imgmin = np.min(image, axis=(0, 1))
        imgmax = np.max(image, axis=(0, 1))

        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.99, saturation=0.9, hue=0., p=0.5)

        ])
        img = train_trans(image=image)['image']

        imgn = img
        imgori = img

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)


        return imgori, img
    def transforms_test(self, image):
        imgmin = np.min(image, axis=(0, 1))
        imgmax = np.max(image, axis=(0, 1))
        img = 1 * ((image - imgmin) / (imgmax - imgmin + 1e-10))
        image = np.array(img, dtype=np.float32)

        train_trans = A.Compose([
        ])

        img = image

        imgn = img
        imgori = img

        imgori = imgori.transpose(2, 0, 1)
        img = imgn.transpose(2, 0, 1)
        return imgori, img

    def zhenghe(self, img_dir, idlist, img_dir2, idlist2):
        dom1t2pth = []
        dom1dwipth = []
        dom1adcpth = []
        dom2t2pth = []
        dom2dwipth = []
        dom2adcpth = []
        for d1 in idlist:
            dom1t2pth.append([os.path.join(img_dir, 'T2', d1), 0])
            dom1dwipth.append([os.path.join(img_dir, 'DWI', d1), 0])
            dom1adcpth.append([os.path.join(img_dir, 'ADC', d1), 0])
        for d2 in idlist2:
            dom2t2pth.append([os.path.join(img_dir2, 'T2', d2), 1])
            dom2dwipth.append([os.path.join(img_dir2, 'DWI', d2), 1])
            dom2adcpth.append([os.path.join(img_dir2, 'ADC', d2), 1])
        domt2pth = dom1t2pth + dom2t2pth
        domdwipth = dom1dwipth + dom2dwipth
        domadcpth = dom1adcpth + dom2adcpth

        return domt2pth, domdwipth, domadcpth

if __name__ == '__main__':

    root = '/data_raid5_21T/fuxu/datasets/psm_dataset/psmdataset/train_test_h5_2d-cli4/D2-sanyuan'
    root2 = '/data_raid5_21T/fuxu/datasets/psm_dataset/psmdataset/train_test_h5_2d-cli4/D1-fuyiyuan'

    ids = os.listdir(os.path.join(root, 'T2'))
    ids2 = os.listdir(os.path.join(root2, 'T2'))
    print(ids)

    mydata = Datasetdom4v2(root, ids, root2, ids2,  phase='train')

    mydataloader1 = DataLoader(dataset=mydata, batch_size=64, shuffle=False)
    print(len(mydataloader1))
    for data in mydataloader1:
        id, imgori, img, label, cli = data
        label = label.long()
        labelsn = torch.flip(label, dims=(0,))
        duiy = label ^ labelsn
        print(duiy, torch.sum(duiy))



