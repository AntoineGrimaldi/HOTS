from network import network
from tools import fit_MLR, predict_MLR, score_classif_events, score_classif_time, get_loader, fit_histo, predict_histo
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def online_accuracy(network, tau_cla, trainset_raw, testset_raw, testset_tstpms, date, timestep, thres=None, width_fig = 20):

    model, loss = fit_MLR(tau_cla, network = network, verbose = False)
    likelihood, true_target, timestamps = predict_MLR(model, tau_cla, network = network, verbose = False)
    _, onlinac_ev, _, _, _ = score_classif_events(likelihood, true_target, verbose=False)
    _, onlinac_time, _, _, _ = score_classif_time(likelihood, true_target, timestamps, timestep, thres=thres, verbose=False)
    
    model_raw, loss_raw = fit_MLR(tau_cla, date = date, dataset_as_input = trainset_raw, verbose = False)
    likelihood_raw, true_target_raw, timestamps_raw = predict_MLR(model_raw, tau_cla, date = date, dataset_as_input = testset_raw, dataset_for_timestamps_as_input = testset_tstpms, verbose = False)
    _, onlinac_ev_raw, _, _, _ = score_classif_events(likelihood_raw, true_target_raw, verbose=False)
    _, onlinac_time_raw, _, _, _ = score_classif_time(likelihood_raw, true_target_raw, timestamps_raw, timestep, thres=thres, verbose=False)
    
    fig, axs = plt.subplots(1,2, figsize=(width_fig,width_fig/3))
    axs[0].semilogx(onlinac_ev, '.');
    axs[0].semilogx(onlinac_ev_raw, '.');
    axs[0].set_xlabel('number of events');
    axs[0].set_ylabel('online accuracy');
    axs[0].set_title('LR classification results evolution as a function of the number of events');
    axs[1].semilogx(onlinac_time, '.', label='online hots');
    axs[1].semilogx(onlinac_time_raw, '.', label='raw event stream');
    axs[1].set_xlabel('time (in ms)');
    axs[1].set_ylabel('online accuracy');
    axs[1].set_title('LR classification results evolution as a function of time');
    axs[1].legend()
    
def clustering_variability(trainset, testset, homeo, tau, date, nb_trials=100):
    
    nb_class = len(trainset.classes)
    sensor_size = trainset.sensor_size
    train_loader = get_loader(trainset)
    test_loader = get_loader(testset)
    acc = []
    hom_acc = []
    acc_3 = []
    hom_acc_3 = []
    acc_6 = []
    hom_acc_6 = []
    pbar = tqdm(total=nb_trials)
    for trial in range(nb_trials):
        timestr = date+f'_{trial}'
        name = 'hots'
        hots = network(name = name, tau = tau, homeo = homeo, timestr = timestr, camsize=(sensor_size[0], sensor_size[1]))
        hots.running(train_loader, trainset.ordering, trainset.classes, learn=True, train=True, verbose=False)
        hots.running(train_loader, trainset.ordering, trainset.classes, learn=False, train=True, verbose=False)
        hots.running(test_loader, trainset.ordering, trainset.classes, learn=False, train=False, verbose=False)
        histo, label = fit_histo(hots, verbose = False)
        acc.append(predict_histo(hots, histo, label, k=1, verbose = False))
        acc_3.append(predict_histo(hots, histo, label, k=3, verbose = False))
        acc_6.append(predict_histo(hots, histo, label, k=6, verbose = False))
        
        name = 'homhots'
        homhots = network(name = name, tau = tau, homeo = homeo, timestr = timestr, camsize=(sensor_size[0], sensor_size[1]))
        homhots.running(train_loader, trainset.ordering, trainset.classes, learn=True, train=True, verbose=False)
        homhots.running(train_loader, trainset.ordering, trainset.classes, learn=False, train=True, verbose=False)
        homhots.running(test_loader, trainset.ordering, trainset.classes, learn=False, train=False, verbose=False)
        histo, label = fit_histo(homhots, verbose = False)
        hom_acc.append(predict_histo(homhots, histo, label, k=1, verbose = False))
        hom_acc_3.append(predict_histo(homhots, histo, label, k=3, verbose = False))
        hom_acc_6.append(predict_histo(homhots, histo, label, k=6, verbose = False))
        pbar.update(1)
    pbar.close()
        

    labels = ['original HOTS', 'HOTS with homeostasis']

    kNN1_means = [np.mean(acc), np.mean(hom_acc)]
    kNN3_means = [np.mean(acc_3), np.mean(hom_acc_3)]
    kNN6_means = [np.mean(acc_6), np.mean(hom_acc_6)]

    kNN1_quant = [[np.quantile(acc, 0.05),np.quantile(acc, 0.95)], [np.quantile(hom_acc, 0.05),np.quantile(hom_acc, 0.95)]]
    kNN3_quant = [[np.quantile(acc_3, 0.05),np.quantile(acc_3, 0.95)], [np.quantile(hom_acc_3, 0.05),np.quantile(hom_acc_3, 0.95)]]
    kNN6_quant = [[np.quantile(acc_6, 0.05),np.quantile(acc_6, 0.95)], [np.quantile(hom_acc_6, 0.05),np.quantile(hom_acc_6, 0.95)]]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects2 = ax.bar(x - width, kNN1_means, width, label='1-NN')
    rects3 = ax.bar(x, kNN3_means, width, label='3-NN')
    rects4 = ax.bar(x + width, kNN6_means, width, label='6-NN')

    for i in range(len(x)):
        ax.plot([x[i] - width,x[i] - width], kNN1_quant[i], '-k')
        ax.plot([x[i],x[i]], kNN3_quant[i], '-k')
        ax.plot([x[i]+width,x[i]+width], kNN6_quant[i], '-k')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.axhline(1/nb_class*100, 0, 1, linestyle='--', color='k', label='chance level')
    ax.set_ylabel('Accuracy (in %)', fontsize=16)
    ax.set_title('Classification performances', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    leg = ax.legend(loc='upper center',fontsize=12)
    leg.get_frame().set_alpha(0)

    fig.tight_layout()

    plt.show()

