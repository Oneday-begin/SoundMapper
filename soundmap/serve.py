"""Single-process Windows-friendly server entry point."""
import asyncio
import sys
import uvicorn

def main():
    if sys.platform=='win32':
        # Avoid Python 3.12 Proactor shutdown noise when players cancel byte-range requests.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run('soundmap.app:app',host='127.0.0.1',port=8765,workers=1,loop='asyncio',access_log=False)

if __name__=='__main__':main()
