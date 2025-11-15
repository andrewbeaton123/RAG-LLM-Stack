import sys
import os
import json
import argparse

# ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_service.app.clients import LMStudioLLM
from langchain.schema import LLMResult  # optional check


def format_output(result):
    # handle possible return shapes (string or LLMResult-like)
    if isinstance(result, str):
        return result
    if hasattr(result, "generations"):
        try:
            gens = result.generations
            # flatten first generation(s)
            return "\n".join(
                str(item[0].text if hasattr(item[0], "text") else item[0]) for item in gens
            )
        except Exception:
            return repr(result)
    return repr(result)


def main():
    p = argparse.ArgumentParser(description="Temporary CLI to test LMStudioLLM")
    p.add_argument("--base-url", default="http://192.168.2.100:1234", help="LM Studio base url")
    p.add_argument("--model", default="deepseek-r1-distill-qwen-14b", help="Model name to use")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--mode", choices=["generate", "chat"], default="generate")
    p.add_argument("--prompt", help="Prompt string for generate; if omitted read from stdin")
    p.add_argument("--messages", help="JSON array of messages for chat e.g. '[{\"role\":\"user\",\"content\":\"Hi\"}]'")
    p.add_argument("--skip-verify", action="store_true", help="Skip verify_connection by setting TESTING=TRUE")
    args = p.parse_args()

    if args.skip_verify:
        os.environ["TESTING"] = "TRUE"

    init_kwargs = {}
    if args.model:
        init_kwargs["model"] = args.model
    if args.temperature is not None:
        init_kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        init_kwargs["max_tokens"] = args.max_tokens

    llm = LMStudioLLM(base_url=args.base_url, **init_kwargs)

    if args.mode == "generate":
        prompt = args.prompt
        if not prompt:
            print("Enter prompt (Ctrl-D to finish):")
            prompt = sys.stdin.read().strip()
        print("Sending prompt...")
        result = llm.generate(prompt)
        print("\n=== RESULT ===")
        print(format_output(result))

    else:  # chat
        if args.messages:
            messages = json.loads(args.messages)
        else:
            print("Enter chat messages as JSON array (e.g. [{\"role\":\"user\",\"content\":\"Hi\"}]) or leave blank to enter one message:")
            raw = sys.stdin.read().strip()
            if raw:
                messages = json.loads(raw)
            else:
                usr = input("User message: ")
                messages = [{"role": "user", "content": usr}]
        print("Sending chat...")
        result = llm.chat(messages)
        print("\n=== RESULT ===")
        print(format_output(result))


if __name__ == "__main__":
    main()