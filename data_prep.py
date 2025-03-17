import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to NDPI (or SVS) data asset.")
    parser.add_argument("--output_path", type=str, help="Where to store prepped data.")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    # Placeholder for real data prep (tiling, etc.)
    with open(os.path.join(args.output_path, "data_prep_done.txt"), "w") as f:
        f.write("Data prep completed.\n")
    
    print("Data prep step done. Output at:", args.output_path)

if __name__ == "__main__":
    main()
