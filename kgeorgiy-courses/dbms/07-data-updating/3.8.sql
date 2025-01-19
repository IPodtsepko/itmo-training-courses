update
    Students
set
    Marks = (
        select
            count(Mark)
        from
            Marks
        where
            Marks.StudentId = Students.StudentId
    ),
    Debts = (
        select
            count(distinct CourseId)
        from
            Students S
            left join Plan using (GroupId)
            left join Marks using (StudentId, CourseId)
        where
            S.StudentId = Students.StudentId
            and Mark is null
    );
