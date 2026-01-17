# Quick Start Guide - Enhanced Hybrid Strategy

## 🚀 Quick Test (5 minutes)

Test just the Hybrid strategy:

```bash
cd /Users/dinukaperera/FLwithFlwr/flowertry
conda activate flower_tutorial
python main.py strategy=hybrid num_rounds=20
```

## 📊 Full Comparison (25 minutes)

Compare all strategies (FedAvg, FedProx, SCAFFOLD, Hybrid):

```bash
./test_hybrid.sh
# OR
python main.py compare_all=true num_rounds=45
```

## 🎯 What to Look For

### Success Indicators:

✅ Hybrid RMSE < FedAvg RMSE by 15-30%  
✅ Hybrid R² > FedProx R² by 10-20%  
✅ Hybrid converges faster (steeper drop in early rounds)  
✅ Hybrid curves are smooth (no oscillations)

### Expected Final Metrics:

| Strategy   | RMSE          | R²            | MAPE       |
| ---------- | ------------- | ------------- | ---------- |
| FedAvg     | 3500-4000     | 0.75-0.80     | 18-22%     |
| FedProx    | 3000-3500     | 0.80-0.85     | 15-18%     |
| SCAFFOLD   | 3200-3800     | 0.78-0.83     | 16-20%     |
| **Hybrid** | **2500-3000** | **0.85-0.90** | **12-15%** |

## 🔧 Troubleshooting

### If Hybrid doesn't outperform:

1. **Check rounds**: Need at least 35-45 rounds for full effect

   ```bash
   python main.py strategy=hybrid num_rounds=60
   ```

2. **Verify adaptive weights**: Check console for progressive weight logs

3. **Increase learning rate**: If too slow to converge

   ```bash
   python main.py strategy=hybrid strategy_configs.hybrid.lr=0.001
   ```

4. **More local epochs**: If clients need more training
   ```bash
   python main.py strategy=hybrid strategy_configs.hybrid.local_epochs=5
   ```

## 📈 Key Files Changed

- **[client.py](client.py)**: Enhanced `_train_hybrid()` with adaptive mechanisms
- **[model.py](model.py)**: Deeper network (160→96→48)
- **[conf/base.yaml](conf/base.yaml)**: Optimized Hybrid parameters

## 📖 Detailed Documentation

See [HYBRID_IMPROVEMENTS.md](HYBRID_IMPROVEMENTS.md) for:

- Complete technical explanation
- Mathematical formulation
- Design philosophy
- Comparison tables

## 💡 Quick Tips

- **First run**: Use `num_rounds=20` to verify it works
- **Full evaluation**: Use `num_rounds=45` for best results
- **Debug mode**: Add `verbose=true` to see detailed logs
- **Save results**: Automatically saved in `outputs/YYYY-MM-DD/HH-MM-SS/`

## 🎓 Key Innovations

1. **Progressive Weights**: FedProx-heavy → SCAFFOLD-heavy
2. **Momentum Control**: 0.9 momentum on SCAFFOLD corrections
3. **Adaptive Learning Rate**: 1.5× boost in early rounds
4. **Dynamic Mu**: 0.08 → 0.03 over training
5. **Deeper Model**: 20K parameters vs 10K baseline

---

**Ready to test?** Run: `./test_hybrid.sh`
