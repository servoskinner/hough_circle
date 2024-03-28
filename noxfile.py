import nox

nox.options.sessions = "lint", "tests"
lint_targets = "src", "test.py", "noxfile.py"


@nox.session()
def lint(session):
    args = session.posargs or lint_targets
    session.install("black")
    session.run("black", *args)


@nox.session()
def test(session):
    session.run("poetry", "install", external=True)
    session.run()
    session.run("pytest", "test.py")
