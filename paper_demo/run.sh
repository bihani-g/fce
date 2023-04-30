for dataset in "news" "agnews" "imdb"

do
    for size in 100 1000 5000 10000
        do
            echo "Running analysis for $dataset $size "
            python get_predictions.py $dataset $size "demo_data" "demo_results"
            python binning.py $dataset $size 15 "demo_results"
        done
done
       