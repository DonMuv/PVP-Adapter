import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# from models.networks2 import *
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
import cv2
import wandb
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

vis_label_list = []
vis_pred_list = []

# wandb.init(config=all_args,
#                project=your_project_name,
#                entity=your_team_name,
#                notes=socket.gethostname(),
#                name=your_experiment_name
#                dir=run_dir,
#                job_type="training",
#                reinit=True)


# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            print('===> Loading val checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            print('===> Loaded val checkpoint!')

            self.net_G.to(self.device)

            # 计算参数量
            total_params = sum(p.numel() for p in self.net_G.parameters())
            print(f"======= Total parameters: {total_params / 1e6:.2f} M")

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self, batch):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        # # 保存测试集的所有预测图
        each_name = batch['name']
        each_vis_pred = utils.make_numpy_grid(self._visualize_pred())
        # each_save_dir = os.path.join(self.vis_dir, 'pred')
        each_save_dir = 'vis/20250523_val1'
        os.makedirs(each_save_dir, exist_ok=True)
        
        each_file_name = os.path.join(each_save_dir, each_name[0])
        # print(np.unique(each_vis_pred))
        cv2.imwrite(each_file_name, each_vis_pred)
        # sImage = Image.fromarray(np.uint8(each_vis_pred))
        # sImage.save(each_file_name)
        
        # plt.imsave(each_file_name, each_vis_pred)
        ###

        # # 生成所有热力图
        # base_image_path = os.path.join('../PVPanel-CD/val/B', each_name[0])
        # base_image = cv2.imread(base_image_path)
        # base_image = cv2.resize(base_image, (512, 512), cv2.INTER_LINEAR)
        # base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        # heat_pred = self.G_pred.cpu().numpy()
        # # print('=====', heat_pred.shape)
        # heat_pred = heat_pred[:,1,:,:][0]
        # # heat_pred = np.expand_dims(heat_pred, axis=2)

        # heat_norm = (heat_pred - heat_pred.min()) / (heat_pred.max() - heat_pred.min())
        # heat_pred = (1-heat_norm) * 255
        # heat_pred = heat_pred.astype(np.uint8)
        # heatmap = cv2.applyColorMap(heat_pred, cv2.COLORMAP_SUMMER)
        # # overlay = cv2.addWeighted(base_image, 0.5, heatmap, 0.5, 0)
        # over_dir = os.path.join(self.vis_dir, 'overlay')
        # os.makedirs(over_dir, exist_ok=True)
        # each_over_name = os.path.join(over_dir, each_name[0])
        # cv2.imwrite(each_over_name, overlay)
        
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, 100) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)

        # new
        # fx1, fx2, cp = self.net_G(img_in1, img_in2)
        # self.G_pred = cp[-1]
        self.G_pred = self.net_G(img_in1, img_in2)[-1]

        # 可视化特征空间
        # print('111', self.G_pred.cpu().numpy().shape)   # (1, 2, 512, 512)
        # print('222', batch['L'].cpu().numpy().shape)   # (1, 1, 512, 512)
        vis_pred_list.append(self.G_pred.cpu().numpy())
        vis_label_list.append(batch['L'].cpu().numpy())

        # # 可视化特征图
        # visualizer = Visualizer()
        # print('===========', fx1[0].shape)
        # vis_feat1 = fx1[0].squeeze(0)
        # vis_feat1 = vis_feat1.mean(dim=0, keepdim=True)
        # vis_feat1 = vis_feat1.repeat(3, 1, 1)
        # print(vis_feat1)
        # vis_feat1 = vis_feat1.permute(1,2,0)*255
        # print(vis_feat1)
        # vis_feat1 = vis_feat1.cpu().numpy().astype(np.uint8)
        # print(vis_feat1)
        
        # # visualizer.show(visualizer.draw_featmap(vis_feat1, channel_reduction='squeeze_mean'))
        # file_name = os.path.join(
        #         self.vis_dir, 'featmap', 'eval_' + str(self.batch_id)+'.jpg')
        # plt.imsave(file_name, vis_feat1)
        # 停
        
        # vis_feat1 = fx1[0].cpu().numpy()[:,:,:3] * 255
        # vis_feat2 = (fx1[1].permute(0, 2, 3, 1).float()[0].detach().cpu().numpy()*255).astype(np.uint8)[:,::3]
        # vis_feat3 = (fx1[2].permute(0, 2, 3, 1).float()[0].detach().cpu().numpy()*255).astype(np.uint8)[:,::3]
        # vis_feat4 = (fx1[3].permute(0, 2, 3, 1).float()[0].detach().cpu().numpy()*255).astype(np.uint8)[:,::3]
        # file_name = os.path.join(
        #         self.vis_dir, 'featmap', 'eval_' + str(self.batch_id)+'.jpg')
        # plt.imsave(file_name, vis_feat1)
        # 停
        

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states(batch)   # 已修改
            # if self.batch_id == 1000:
            #     break
        self._collect_epoch_states()

        # new
        # sne_pred = np.concatenate(vis_pred_list, axis=0)
        # sne_label = np.concatenate(vis_label_list, axis=0).astype(np.float32)
        # np.save('sne_pred.npy', sne_pred)
        # np.save('sne_label.npy', sne_label)

        # tsne_plot('./', sne_label, sne_pred)
        
