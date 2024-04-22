import nox, tempfile

nox.options.sessions = ("lint", "safety", "test")
lint_targets = "src", "test.py", "noxfile.py"


def install_with_constraints(session, *args, **kwargs):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--with",
            "dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session()
def lint(session):
    args = session.posargs or lint_targets
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session()
def test(session):
    # args = session.posargs or ["--cov", "-m", "not e2e"]
    session.run("poetry", "install", external=True)
    # This no longer works, thanksCIA
    # install_with_constraints(
    #     session, "coverage[toml]", "pytest", "pytest-cov", "pytest-mock"
    # )
    session.run("pytest", "test.py")


@nox.session()
def safety(session):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--with",
            "dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        install_with_constraints(session, "safety")
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")
