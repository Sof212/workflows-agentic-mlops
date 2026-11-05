import docker
import tempfile
import os
import uuid

class agents_verify_code:
    def __init__(self):
        pass
        

    def execute_python_code(
        code: str,
        requirements: list = None,
        timeout: int = 60
    ):
        client = docker.from_env()
        container = None
        volume = None

        requirements = requirements or []
        image = "python:3.11-slim"

        try:
            # 1. Check Docker image
            try:
                client.images.get(image)
                print(f"Image {image} already available.")
            except docker.errors.ImageNotFound:
                client.images.pull(image)

            # 2. Create temporary files
            with tempfile.TemporaryDirectory() as tmpdir:

                with open(os.path.join(tmpdir, "main.py"), "w") as f:
                    f.write(code)

                with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
                    f.write("\n".join(requirements))

                # 3. Create temporary Docker volume
                volume_name = f"python-deps-{uuid.uuid4().hex[:12]}"

                volume = client.volumes.create(
                    name=volume_name
                )

                # 4. Run container
                command = """
    set -e

    pip install \
        --no-cache-dir \
        --disable-pip-version-check \
        --target=/deps \
        -r /app/requirements.txt

    PYTHONPATH=/deps python -u /app/main.py
    """

                container = client.containers.run(
                    image=image,
                    command=["sh", "-c", command],

                    volumes={
                        tmpdir: {
                            "bind": "/app",
                            "mode": "ro"
                        },
                        volume.name: {
                            "bind": "/deps",
                            "mode": "rw"
                        }
                    },

                    working_dir="/app",

                    mem_limit="1g",
                    nano_cpus=1000000000,
                    pids_limit=64,

                    read_only=True,
                    network_disabled=False,

                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges:true"],

                    tmpfs={
                        "/tmp": "rw,noexec,nosuid,size=256m"
                    },

                    detach=True
                )

                # 5. Wait with timeout
                try:
                    result = container.wait(timeout=timeout)

                except Exception as e:
                    try:
                        container.kill()
                    except docker.errors.DockerException:
                        pass

                    return {
                        "exit_code": -1,
                        "output": f"Execution timeout: {str(e)}"
                    }

                # 6. Retrieve logs
                logs = container.logs().decode(
                    "utf-8",
                    errors="replace"
                )

                return {
                    "exit_code": result["StatusCode"],
                    "output": logs
                }

        except Exception as e:
            return {
                "exit_code": -1,
                "output": f"Docker error: {str(e)}"
            }

        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except docker.errors.DockerException:
                    pass

            if volume is not None:
                try:
                    volume.remove(force=True)
                except docker.errors.DockerException:
                    pass

            client.close()


