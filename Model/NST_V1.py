import torch
import torch.optim as optim
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torchvision import models
import os
# VGG19 model according to the paper

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19,self).__init__()
        self.layers = ['0','5','10','19','28']
        self.model = models.vgg19(pretrained=True,progress=True).features[:29]

    def forward(self,x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
           # print(x)
            if str(layer_num) in self.layers:
                features.append(x)
        return features

# for loading the image
def loadimage(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


# for using dedicate gpu in the system
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image size
size_of_image = 400

# As image can of different size we are just resizing the them to make their size equal
loader = transforms.Compose(
        {
            transforms.Resize((2*size_of_image, size_of_image)),
            transforms.ToTensor()
        }
)

# importing style and content images
content_image = loadimage('content.jpg')
style_image = loadimage("image.jpg")

# their are two choices either you can use a random image or using content image gives better results
#generated = torch.randn(content_image.shape, device = device ,requires_grad = True)
generated = content_image.clone().requires_grad_(True)

# metrics for calculating loss function  and number of epochs

No_of_Iterations = 1000
learning_rate = 0.001
# try to keep alpha/beta = 10^-3 for good results
alpha = 1.0
beta = 0.001
# importing the pretrained VGG 19 model
model = VGG19().to(device).eval()


# loss function to construct the image
def lossfunction(content_image, style_image, model):
    # extracting feature maps from the model for the images
    generated_features = model(generated)
    original_features = model(content_image)
    style_features = model(style_image)
    # intializing the losses
    style_loss = content_loss = 0.
    # Calculation of the Total loss value
    for gen_feature, content_feature, style_feature in zip(generated_features, original_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        content_loss += torch.mean((gen_feature - content_feature) ** 2)

        # Gram matrix formation for calculating style loss
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )

        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        style_loss += torch.mean((G - A) ** 2)
    # total is a linear combination of style and content loss according to the research paper
    total_loss = alpha * content_loss + beta * style_loss
     # returns total_loss
    return total_loss, style_loss, content_loss

# Paper suggests LGBFS optimiser but it works with Adam as well
optimizer = optim.Adam([generated],lr = learning_rate)

#model = VGG19().to(device).eval()
# running given number of epochs
for i in range(1,No_of_Iterations+1):
  total_loss , style_loss,content_loss = lossfunction(content_image,style_image,model)
  optimizer.zero_grad()
  total_loss.backward() # changing the value of the image
  optimizer.step()
  print("Loss Value",total_loss)
  out = "output_image"+str(i)+".jpeg"
  save_image(generated,out)   # Storing the output image
