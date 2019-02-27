import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, train_loader, val_loader, fail_tol, learning_rate=3e-4, label=""):

    num_epochs = 100

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)
    
    total_step = len(train_loader)
    loss_list, val_acc_list = [], []
    
    fail_cnt, cur_best = 0, 0
    for epoch in range(num_epochs):
        
        avg_loss = 0
        
        for i, (data, lengths, ind_sort, ind_unsort, labels) in enumerate(train_loader):

            model.train()
            optimizer.zero_grad()

            outputs = model(data, ind_sort, ind_unsort, lengths)
            obj = criterion(outputs, labels)
            obj.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            avg_loss += obj.item() / len(train_loader)
            
        val_acc = test_model(val_loader, model)
        val_acc_list.append(val_acc)
        loss_list.append(avg_loss)

        if (val_acc > cur_best):
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4}, Val: {:.4}'.format(
                epoch+1, num_epochs, i+1, len(train_loader), avg_loss, val_acc))
            print("found best! save model...")
            torch.save(model.state_dict(), 'model' + "-" + label + '.ckpt')
            print("model saved")
            cur_best = val_acc
            fail_cnt = 0
        else:
            fail_cnt += 1
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4}, Val: {:.4} ({}/{})'.format(
                epoch+1, num_epochs, i+1, len(train_loader), avg_loss, val_acc, fail_cnt, fail_tol))
        if fail_cnt > fail_tol:
            return loss_list, val_acc_list

#         scheduler.step(val_acc)
    return loss_list, val_acc_list

def test_model(loader, model):
    from collections import Counter
    res_cnt = Counter()
    correct = 0
    total = 0
    model.eval()
    for data, lengths, ind_sort, ind_unsort, labels in loader:
        outputs = F.softmax(model(data, ind_sort, ind_unsort, lengths), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        
        res_cnt.update(list(predicted.squeeze().cpu().numpy()))
        
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    print(res_cnt)
    return (100 * correct / total)
