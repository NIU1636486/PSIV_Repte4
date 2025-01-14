
""" 
En canvi de en el use_groups: 
kf = GroupKFold(n_splits=num_folds)  # Ensures patient-level split
splits = kf.split(X, y, groups=groups)

fer leave one patient out logic: 
unique_patients = list(set(groups))  # Extract unique patient IDs
splits = []

for test_patient in unique_patients:
    train_idx = [i for i, patient in enumerate(groups) if patient != test_patient]
    test_idx = [i for i, patient in enumerate(groups) if patient == test_patient]
    splits.append((train_idx, test_idx))  # Store as (train, test) tuple


"""
# Training loop
for fold, (train_idx, test_idx) in enumerate(splits):
    print(f"\nFold {fold + 1}: Leaving Out Patient {unique_patients[fold]} for Testing")

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    inputmodule_params, net_params, outmodule_params = get_default_hyperparameters()
    model = EpilepsyLSTM(inputmodule_params, net_params, outmodule_params)
    model.init_weights()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}")

    # Evaluate the model on the unseen patient
    print("Starting validation...")
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == y_batch).sum().item()
            test_total += y_batch.size(0)

    test_accuracy = test_correct / test_total
    print(f"Test Patient {unique_patients[fold]} - Accuracy: {test_accuracy:.4f}, Loss: {test_loss / len(test_loader):.4f}")
