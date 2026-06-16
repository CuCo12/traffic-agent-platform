import os
import urllib.request

# Dictionary of key papers on traffic signal control and core AI methods from arXiv
PAPERS = {
    "Attention_Is_All_You_Need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "DQN_Playing_Atari.pdf": "https://arxiv.org/pdf/1312.5602.pdf",
    "CoLight_Traffic_Signal_Control.pdf": "https://arxiv.org/pdf/1909.13034.pdf",
    "PressLight_Max_Pressure_Control.pdf": "https://arxiv.org/pdf/1906.07228.pdf",
    "Toward_A_Thousand_Lights.pdf": "https://arxiv.org/pdf/1909.05714.pdf",
    "TransformerLight.pdf": "https://arxiv.org/pdf/2308.10543.pdf"
}

def main():
    papers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")
    os.makedirs(papers_dir, exist_ok=True)
    print(f"📁 Target directory for papers: {papers_dir}\n")
    
    for filename, url in PAPERS.items():
        dest = os.path.join(papers_dir, filename)
        if os.path.exists(dest):
            print(f"✅ {filename} already exists. Skipping download.")
            continue
            
        print(f"📥 Downloading {filename} from {url}...")
        try:
            # Use a browser User-Agent to prevent HTTP 403 Forbidden from arXiv CDN
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(dest, 'wb') as out_file:
                    out_file.write(response.read())
            print(f"✨ Successfully downloaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            
    print("\n🎉 Paper downloading task completed! You can now run the RAG literature analysis.")

if __name__ == "__main__":
    main()
