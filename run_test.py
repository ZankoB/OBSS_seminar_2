import numpy as np
from database_reader import database_reader
from qrs_classifier import qrs_classifier

def run_test():
    reader = database_reader()
    records = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
               111, 112, 113, 114, 115, 116, 117, 118, 119,
               121, 122, 123, 124, 200, 201, 202, 203, 205,
               207, 208, 209, 210, 212, 213, 214, 215, 217,
               219, 220, 221, 222, 223, 228, 230, 231, 232,
               233, 234]
    
    # Global Counters
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    for rec in records:
        print(f"\nProcessing Record {rec}...")
        try:
            data = reader.get_data(rec)
        except:
            continue

        clf = qrs_classifier(fs=data["fs"])
        qrs, labels, positions = clf.extract_qrs(data["signal"], data["annotations_samples"], data["annotations_symbols"])
        
        if len(qrs) == 0: continue

        qrs = np.array([clf.normalize_qrs(q) for q in qrs])
        clf.build_reference(qrs, labels)
        clf.estimate_threshold(qrs, labels) 
        pred = clf.classify(qrs, positions)

        # Update Global Counts
        TP = np.sum((labels == 'V') & (pred == 'V'))
        TN = np.sum((labels == 'N') & (pred == 'N'))
        FP = np.sum((labels == 'N') & (pred == 'V'))
        FN = np.sum((labels == 'V') & (pred == 'N'))

        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN
        
        # Print local result for debugging
        Se = TP / (TP + FN + 1e-10)
        Pp = TP / (TP + FP + 1e-10)
        f1 = 2 * Se * Pp / (Se + Pp) if (Se + Pp) > 0 else 0
        print(f"-> Local F1: {f1:.4f}")

    # --- CALCULATE GLOBAL METRICS ---
    print("\n" + "="*50)
    print("FINAL GLOBAL PERFORMANCE (Pooled)")
    print("="*50)
    
    Global_Se = total_TP / (total_TP + total_FN)
    Global_Pp = total_TP / (total_TP + total_FP)
    Global_Sp = total_TN / (total_TN + total_FP)
    Global_F1 = 2 * Global_Se * Global_Pp / (Global_Se + Global_Pp)

    print(f"Total PVCs Detected (TP): {total_TP}")
    print(f"Total Missed PVCs (FN):   {total_FN}")
    print(f"Total False Alarms (FP):  {total_FP}")
    print("-" * 30)
    print(f"Global Sensitivity: {Global_Se:.4f}")
    print(f"Global Precision:   {Global_Pp:.4f}")
    print(f"Global Specificity: {Global_Sp:.4f}")
    print(f"GLOBAL F1 SCORE:    {Global_F1:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_test()