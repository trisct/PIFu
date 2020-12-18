dataset_name="normalized_dataset"

for dataname in $dataset_name/*
do
	echo "Running python -m apps.render_data -i $dataname -o training_dataset -p"
	python -m apps.render_data -i $dataname -o training_dataset -p
done
