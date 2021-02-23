import inspect
import uuid
from opencensus.trace.tracer import Tracer


class Severity:
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LoggerInterface(Tracer):

    def log_metric(self, name, value, description, log_parent):
        pass

    def log(self, name, value, description, severity, log_parent):
        pass

    def exception(self, exception):
        pass

    def span(self, name='span'):
        """Create a new span with the trace using the context information.
        :type name: str
        :param name: The name of the span.
        :rtype: :class:`~opencensus.trace.span.Span`
        :returns: The Span object.
        """
        pass

    def start_span(self, name='span'):
        """Start a span.
        :type name: str
        :param name: The name of the span.
        :rtype: :class:`~opencensus.trace.span.Span`
        :returns: The Span object.
        """
        pass

    def end_span(self):
        """End a span. Remove the span from the span stack, and update the
        span_id in TraceContext as the current span_id which is the peek
        element in the span stack.
        """
        pass

    def current_span(self):
        """Return the current span."""
        pass

    def add_attribute_to_current_span(self, attribute_key, attribute_value):
        pass

    def list_collected_spans(self):
        """List collected spans."""
        pass


class ObservabilityAbstract:
    OFFLINE_RUN = "OfflineRun"
    CUSTOM_DIMENSIONS = "custom_dimensions"
    CORRELATION_ID = "correlation_id"
    FILENAME = "fileName"
    MODULE = "module"
    PROCESS = "process"
    LINENO = "lineNumber"
    severity = Severity()
    severity_map = {10: "DEBUG", 20: "INFO",
                    30: "WARNING", 40: "ERROR", 50: "CRITICAL"}

    def get_run_id_and_set_context(self, run):
        """
        gets the correlation ID by the in following order:
        - If the script is running  in an Online run Context of AML --> run_id
        - If the script is running where a build_id
        environment variable  is set --> build_id
        - Else --> generate a unique id

        Sets also the custom context dimensions based on On or Offline run
        :param run:
        :return: correlation_id
        """
        run_id = str(uuid.uuid1())
        if not run.id.startswith(self.OFFLINE_RUN):
            run_id = run.id
            parent_id, portal_url = "none", "none"
            if run.parent is not None:
                parent_id = run.parent.id
                portal_url = run.parent.get_portal_url()
            self.custom_dimensions = {
                'custom_dimensions': {
                    "parent_run_id": parent_id,
                    "step_id": run.id,
                    "step_name": run.name,
                    "experiment_name": run.experiment.name,
                    "run_url": portal_url,
                    "offline_run": False
                }
            }
        elif self.env.build_id:
            run_id = self.env.build_id
            self.custom_dimensions = {
                'custom_dimensions': {
                    "run_id": self.env.build_id,
                    "offline_run": True
                }
            }
        else:
            self.custom_dimensions = {
                'custom_dimensions': {
                    "run_id": run_id,
                    "offline_run": True
                }
            }
        return run_id

    @staticmethod
    def get_callee(stack_level):
        """
        This method get the callee location in [file_name:line_number] format
        :param stack_level:
        :return: string of [file_name:line_number]
        """
        try:
            stack = inspect.stack()
            file_name = stack[stack_level + 1].filename.split("/")[-1]
            line_number = stack[stack_level + 1].lineno
            return "{}:{}".format(file_name, line_number)
        except IndexError:
            print("Index error, failed to log to AzureML")
            return ""

    @staticmethod
    def get_callee_details(stack_level):
        """
        This method returns the callee details as a tuple,
        tuple values ar all strings.
        :param stack_level:
        :return: (module_name, file_name, line_number)
        """
        try:
            stack = inspect.stack()
            file_name = stack[stack_level + 1].filename
            line_number = stack[stack_level + 1].lineno
            module_name = inspect.getmodulename(file_name)
            return module_name, file_name, line_number
        except IndexError:
            print("Index error, failed to log to AzureML")
            return ""
