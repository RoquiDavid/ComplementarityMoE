for seed in 10 20 30 40 50 60 70 80 90 100; do
    echo "================================"
    echo "Testing seed $seed"
    echo "================================"
    
    python ComprehensiveBenchmark.py \
        --data_dir /data/roqui/cmu_mosei_data \
        --results_dir /data/roqui/seed_search/seed_${seed} \
        --num_tasks 2 \
        --batch_size 32 \
        --epochs_per_task 5 \
        --lr 0.001 \
        --seed $seed \
        --tau 0.2 \
        --lambda_barlow 0.01 \
        --lambda_ewc 500 | tee /data/roqui/seed_search/seed_${seed}.log
    
    # Extraire résultat ComplementarityMoE
    grep "ComplementarityMoE_Ours" /data/roqui/seed_search/seed_${seed}/summary.json | head -5
done

echo "Done - Check logs in /data/roqui/seed_search/"
