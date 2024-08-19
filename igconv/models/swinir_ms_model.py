import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.losses import build_loss
from basicsr.utils import imwrite, tensor2img, get_root_logger

from collections import OrderedDict
from tqdm import tqdm
import os.path as osp


@MODEL_REGISTRY.register()
class SwinIRMSModel(SRModel):

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        # scale = self.opt.get('scale', 1)
        scale = self.cur_scale
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_discrete'):
            with torch.no_grad():
                self.net_g_discrete.eval()
                self.output = self.net_g_discrete(img, scale)
        elif hasattr(self, 'net_g_ema'):
            # EMA not wrapped by DDP
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img, scale)
        else:
            self.net_g.eval()
            # with torch.no_grad():
            #     self.output = self.net_g(img, scale)
            if hasattr(self.net_g, 'module'):
                module = self.net_g.module
                with torch.no_grad():
                    self.output = module(img, scale)
            else:
                with torch.no_grad():
                    self.output = self.net_g(img, scale)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # self.cur_scale = data['scale']
        scale = data['scale']
        scale = scale[0] if len(scale) > 1 else scale
        self.cur_scale = int(scale)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.cur_scale)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
                
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict['l_freq'] = l_fft

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            
    def init_training_settings(self):
        self.net_g.train()
        self.use_amp = self.opt.get('use_amp', False)
        if self.use_amp:
            logger = get_root_logger()
            logger.info('Use mixed precision training.')
            self.scaler = torch.cuda.amp.GradScaler()
        
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if train_opt.get('wave_opt'):
            self.cri_wave = build_loss(train_opt['wave_opt']).to(self.device)
        else:
            self.cri_wave = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_freq is None and self.cri_wave is None:
            raise ValueError('Pixel, perceptual and frequency losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        discretize_net_g = self.opt['val'].get('discretize_net_g', False)
        
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            if discretize_net_g and not hasattr(self, 'net_g_discrete'):
                self.net_g_discrete = self.net_g_ema.discretize(self.cur_scale) if hasattr(self, 'net_g_ema')\
                    else self.net_g.discretize(self.cur_scale)
                self.net_g_discrete.eval()
                logger = get_root_logger()
                logger.info(f'Discretize IGConv in net_g for scale {self.cur_scale} ...')
            
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    opt_['crop_border'] = self.cur_scale
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()
            
        if discretize_net_g:
            del self.net_g_discrete

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            