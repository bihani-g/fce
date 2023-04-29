# for dataset in "news" "agnews" "imdb"

# do
#     for size in 100 1000 5000 10000
#         do
#             echo "Running analysis for $dataset $size "
#             python finetune.py $dataset $size
#         done
# done
         
echo "Running analysis for news 100"
python finetune.py "news" 100
