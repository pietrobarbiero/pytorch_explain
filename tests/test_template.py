import unittest


class TestTemplateObject(unittest.TestCase):
    def test_object(self):
        import template

        t = template.TemplateObject()
        self.assertTrue(isinstance(t, template.TemplateObject))
        return


if __name__ == '__main__':
    unittest.main()
