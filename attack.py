import os
import numpy as np

import argparse

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from differential_evolution import differential_evolution

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--model', default='vgg16', help='The target model')
parser.add_argument('--pixels', default=1, type=int,
                    help='The number of pixels that can be perturbed.')
parser.add_argument('--max-examps', default=None, type=int,
                    help='The max number of examples to attack.')
parser.add_argument('--maxiter', default=100, type=int,
                    help='The max number of iterations in the DE algorithm.')
parser.add_argument('--popsize', default=400, type=int,
                    help='The number of adverisal examples in each iteration.')
parser.add_argument('--samples', default=100, type=int,
                    help='The number of image samples to attack.')
parser.add_argument('--targeted', action='store_true',
                    help='Set this switch to test for targeted attacks.')
parser.add_argument('--save', default='./results/results.pkl',
                    help='Save location for the results with pickle.')
parser.add_argument('--verbose', action='store_true',
                    help='Print out additional information every iteration.')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()


def perturb_image(xs, img):
    # TODO - note that "denormaliztion" here uses hardcoded CIFAR10 numbers.
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)

    count = 0
    for x in xs:
        pixels = np.split(x, len(x)/5)

        for pixel in pixels:
            x_pos, y_pos, r, g, b = pixel
            imgs[count, 0, x_pos, y_pos] = (r/255.0-0.4914)/0.2023
            imgs[count, 1, x_pos, y_pos] = (g/255.0-0.4822)/0.1994
            imgs[count, 2, x_pos, y_pos] = (b/255.0-0.4465)/0.2010
        count += 1

    return imgs


def predict_classes(xs, img, target_class, net, minimize=True):
    imgs_perturbed = perturb_image(xs, img.clone())
    # TODO - Variable is deprecated, use torch.from_numpy; by default,
    # `requires_grad` is False.
    input = Variable(imgs_perturbed, volatile=True).cuda() if use_cuda \
        else Variable(imgs_perturbed, volatile=True)
    predictions = F.softmax(net(input)).data.cpu().numpy()[:, target_class]

    return predictions if minimize else 1 - predictions


def attack_success(
        x, img, target_class, net, targeted_attack=False, verbose=False):

    attack_image = perturb_image(x, img.clone())
    # TODO - Variable is deprecated, use torch.from_numpy; by default,
    # `requires_grad` is False.
    input = Variable(attack_image, volatile=True).cuda() if use_cuda \
        else Variable(attack_image, volatile=True)
    confidence = F.softmax(net(input)).data.cpu().numpy()[0]
    predicted_class = np.argmax(confidence)

    if (verbose):
        print("Confidence: %.4f" % confidence[target_class])
    if (targeted_attack and predicted_class == target_class) or \
            (not targeted_attack and predicted_class != target_class):
        return True


def attack(img, label, net, target=None,
           pixels=1, maxiter=75, popsize=400, verbose=False):
    '''
    - img: 1*3*W*H tensor
    - label: a number
    '''

    targeted_attack = target is not None
    target_class = target if targeted_attack else label

    # TODO - bounds are explicitly hardcoded to CIFAR10 images
    bounds = [(0, 32), (0, 32), (0, 255), (0, 255), (0, 255)] * pixels

    popmul = max(1, popsize // len(bounds))

    predict_fn = lambda xs: predict_classes(
        xs, img, target_class, net, target is None)
    callback_fn = lambda x, convergence: attack_success(
        x, img, target_class, net, targeted_attack, verbose)

    inits = np.zeros([popmul*len(bounds), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i*5+0] = np.random.random()*32
            init[i*5+1] = np.random.random()*32
            init[i*5+2] = np.random.normal(128, 127)
            init[i*5+3] = np.random.normal(128, 127)
            init[i*5+4] = np.random.normal(128, 127)

    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul, recombination=1,
        atol=-1, callback=callback_fn, polish=False, init=inits)

    attack_image = perturb_image(attack_result.x, img)
    # TODO - Variable is deprecated, use torch.from_numpy; by default,
    # `requires_grad` is False.
    attack_var = Variable(attack_image, volatile=True).cuda() if use_cuda \
        else Variable(attack_image, volatile=True)
    predicted_probs = F.softmax(net(attack_var)).data.cpu().numpy()[0]

    predicted_class = np.argmax(predicted_probs)

    if (not targeted_attack and predicted_class != label) or \
            (targeted_attack and predicted_class == target_class):
        return 1, attack_result.x.astype(int)
    return 0, [None]


def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400,
               max_examps=None, verbose=False):

    correct = 0
    success = 0

    # TODO - need tqdm here...
    for batch_idx, (input, target) in enumerate(loader):

        if max_examps and batch_idx >= max_examps:
            print("stopping after set number of max examples")
            break

        if verbose:
            print("attacking batch_idx = {}".format(batch_idx))

        # TODO - Variable is deprecated, use torch.from_numpy; by default,
        # `requires_grad` is False.
        img_var = Variable(input, volatile=True).cuda() if use_cuda \
            else Variable(input, volatile=True)
        prior_probs = F.softmax(net(img_var))
        _, indices = torch.max(prior_probs, 1)

        # check whether the model has the prediction correct before attack
        if target[0] != indices.data.cpu()[0]:
            continue

        correct += 1
        target = target.numpy()

        targets = [None] if not targeted else range(10)

        for target_class in targets:
            if (targeted):
                if (target_class == target[0]):
                    continue

            flag, x = attack(input, target[0], net, target_class,
                             pixels=pixels, maxiter=maxiter, popsize=popsize,
                             verbose=verbose)

            success += flag
            if (targeted):
                success_rate = float(success)/(9*correct)
            else:
                success_rate = float(success)/correct

            if flag == 1:
                ps = "success rate: %.4f (%d/%d) [(x,y) = (%d,%d) and " + \
                    "(R,G,B)=(%d,%d,%d)]"
                print(ps % (success_rate, success, correct,
                            x[0], x[1], x[2], x[3], x[4]))

        if correct == args.samples:
            break

    return success_rate


def main():

    print("==> Loading data and model...")
    tranfrom_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        ])
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=tranfrom_test)
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=True, num_workers=2)

    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/%s.t7' % args.model)
    net = checkpoint['net']
    if use_cuda:
        net.cuda()
        cudnn.benchmark = True

    print("==> Starting attck...")

    results = attack_all(net, testloader, pixels=args.pixels,
                         targeted=args.targeted, maxiter=args.maxiter,
                         popsize=args.popsize, max_examps=args.max_examps,
                         verbose=args.verbose)
    print("Final success rate: %.4f" % results)


if __name__ == '__main__':
    main()
